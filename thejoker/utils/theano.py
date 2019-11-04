# Third-party
import theano.tensor as tt
import theano.gof.graph as graph

__all__ = ['get_inputs_of_variable']


def get_inputs_of_variable(variable):
    """
    This function returns the required inputs for input ``TensorVariable``.

    Parameters
    ----------
    variable : ``theano.TensorVariable``

    Returns
    -------
    list
        a list of required inputs to compute the variable.
    """

    variable_inputs = [var for var in graph.inputs(variable)
                       if isinstance(var, tt.TensorVariable)]
    return {p.name: p for p in variable_inputs}
