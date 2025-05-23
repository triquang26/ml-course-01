import sklearn_crfsuite
from sklearn_crfsuite import metrics

class ConditionalRandomField:
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, model_file='trained/crf_model.crfsuite'):
        self.crf = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
            model_filename=model_file
        )