from conditional_random_field import ConditionalRandomField

def train_crf(x_train, y_train, model: ConditionalRandomField) -> ConditionalRandomField:
    print("\nTraining CRF model")
    model.crf.fit(x_train, y_train)
    return model