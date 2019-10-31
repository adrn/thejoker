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
        a list of (tensor variable) to see.
        usally this is a theano function output list. (loss, accuracy, etc.)
    Returns
    -------
    list
        a list of required inputs to compute the variable.
    """

    variable_inputs = [var for var in graph.inputs(variables)
                       if isinstance(var, tt.TensorVariable)]
    return {p.name: p for p in variable_inputs}
