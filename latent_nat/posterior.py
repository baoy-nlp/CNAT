import torch
from torch.autograd import Function

from .code import CategoricalEmbedding as Code
from .code import LearnableCategoricalEmbedding as LearnCode

INF = 1e-5


class SoftCode(LearnCode):
    """
    search the nearest vector by dot-product
    """

    def forward(self, z_e_x):
        """
        :param z_e_x: batch_size, sequence_length, hidden_dim
        :return:
        """
        code = self.embedding.weight
        if not self.update:
            code = code.detach()
        score = dot_product(z_e_x, code)
        indices = score.max(dim=-1)[1]
        return score, indices

    def straight_through(self, z_e_x):
        z_st, indices = straight_through(z_e_x, self.embedding.weight.detach())
        z_st_bar = self.embedding.weight.index_select(dim=0, index=indices)
        z_st_bar = z_st_bar.view_as(z_e_x)
        return z_st, z_st_bar, indices.view(*z_st.size()[:-1]), self.forward(z_e_x)[0]


def dot_product(inputs, codebook):
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.view(-1, embedding_size)
    score = inputs_flatten @ (codebook.t())
    score = score.view(inputs.size(0), inputs.size(1), -1)
    return score


class Search(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            score = dot_product(inputs, codebook)
            indices = torch.max(score, dim=-1)[1]
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class StraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = search(inputs, codebook)
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


search = Search.apply
straight_through = StraightThrough.apply

CODE_CLS = {
    "code": Code,
    "learn-code": LearnCode,
    "soft-code": SoftCode
}
