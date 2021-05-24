import torch
import torch.nn as nn
from torch.autograd import Function

INF = 1e-5


class CategoricalEmbedding(nn.Module):
    def __init__(self, num_codes, code_dim, lamda=0.999, not_update=False, embed_token=None):
        super().__init__()
        self.K = num_codes
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1. / num_codes, 1. / num_codes)
        self.code_count = nn.Parameter(torch.zeros(num_codes).float(), requires_grad=False)
        self.lamda = lamda
        self.update = not not_update

        if embed_token is not None:
            self.cluster_init_code(embed_token)

    def forward_embedding(self, indices):
        embed = self.embedding(indices)
        # if self.update:
        if not self.update:
            embed = embed.detach()
        return embed

    def reinit(self):
        self.code_count = nn.Parameter(torch.zeros(self.K).to(self.code_count), requires_grad=False)

    def forward(self, z_e_x):
        """
        :param z_e_x: batch_size, sequence_length, hidden_dim
        :return:
        """
        indices = vq_search(z_e_x, self.embedding.weight)
        return indices

    def straight_through(self, z_e_x):
        z_st, indices = vq_st(z_e_x, self.embedding.weight.detach())
        z_st_bar = self.embedding.weight.index_select(dim=0, index=indices)
        z_st_bar = z_st_bar.view_as(z_e_x)
        return z_st, z_st_bar, indices.view(*z_st.size()[:-1])

    def cluster_init_code(self, embed_tokens, distance="cosine"):
        from .kmeans import kmeans
        centers = kmeans(
            data=embed_tokens.weight.data,
            num_clusters=self.K,
            distance=distance,
            device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu'),
            initial_state=self.embedding.weight.data
        )[1]
        self.embedding.weight.data = centers.to(self.embedding.weight.data)
        self.embedding.weight.requires_grad = self.update

    def exponential_moving_average(self, enc_sum):
        """

        :param enc_sum: K, D
        :return:
        """
        count_i = self.code_count.view(self.K, -1)  # K,1
        mask = (count_i > 0.0).float()  # K,1
        c = self.embedding.weight.data
        c = mask * (c * self.lamda + (1 - self.lamda) / (count_i + (1 - mask) * INF) * enc_sum) + (1 - mask) * c

        self.embedding.weight.data = c
        self.embedding.weight.requires_grad = self.update

    def forward_count(self, z_e, mask, indices):
        """

        :param z_e: batch_size, sequence_length, D
        :param mask: batch_size, sequence_length
        :param indices: batch_size, sequence_length
        :return:
        """
        mask = mask.long()
        indices = indices * mask - (1 - mask)  # set the masked indices is -1

        inputs_flatten = z_e.view(-1, z_e.size(-1))
        indices_flatten = indices.view(-1)
        enc_sum = []
        for i in range(self.K):
            i_hit = indices_flatten == i  # batch_size*sequence_length,1
            # update assign count
            self.code_count[i] = self.lamda * self.code_count[i] + i_hit.sum().float() * (1 - self.lamda)
            # update embedding
            enc_i_sum = inputs_flatten[i_hit].sum(dim=0)
            enc_sum.append(enc_i_sum)

        return torch.stack(enc_sum)


class LearnableCategoricalEmbedding(CategoricalEmbedding):
    def forward(self, z_e_x):
        embed_size = self.embedding.weight.size(1)
        z_e_flatten = z_e_x.view(-1, embed_size)
        score = z_e_flatten @ (self.embedding.weight.t())

        score = score.view(z_e_x.size(0), z_e_x.size(1), -1)
        indices = score.max(dim=-1)[1]
        return score, indices

    def straight_through(self, z_e_x):
        score, indices = self.forward(z_e_x)
        z_q = self.embedding(indices)
        return z_q, z_q, indices, score


def nearest_search(inputs, codebook):
    embedding_size = codebook.size(1)
    inputs_size = inputs.size()
    inputs_flatten = inputs.view(-1, embedding_size)

    codebook_sqr = torch.sum(codebook ** 2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

    # Compute the distances to the codebook
    distances = torch.addmm(codebook_sqr + inputs_sqr,
                            inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

    return distances


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq_search(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                   .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return grad_inputs, grad_codebook


def search(inputs, codebook):
    embedding_size = codebook.size(1)
    inputs_size = inputs.size()
    inputs_flatten = inputs.view(-1, embedding_size)
    score = inputs_flatten @ (codebook.t())
    _, indices_flatten = score.max(dim=1)
    indices = indices_flatten.view(*inputs_size[:-1])
    return indices


def search_st(inputs, codebook):
    indices = search(inputs, codebook)
    indices_flatten = indices.view(-1)

    codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
    codes = codes_flatten.view_as(inputs)

    return codes, indices_flatten


vq_search = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
