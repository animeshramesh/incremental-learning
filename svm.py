import pegasos
import numpy as np

class SVM_Triplet:
    def __init__(self, X1, X2, Y, base_classes, pos_class, new_class):
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.base_classes = base_classes
        self.pos_class = pos_class
        self.new_class = new_class

def prepare_features(pos_class, neg_classes, feature_vectors, is_train=True, equal_features=False):
    '''
    Returns 1024-dim features for each image which would be used for SVM training
    inputs : 
        is_train -> will return 400 features for each class if is_train=True, else returns 100 features
        equal_features -> if set to True, then len(neg_features) = len(pos_features)
    
    Returns:
        pos_features -> features of images in the positive class
        neg_features -> features of images in the negative classes
    '''
    
    # First 400 images will be used for training. Other 100 for testing
    TRAIN_SPLIT = 400
    
    pos_features = feature_vectors[pos_class]    # 500 x 1024 
    neg_features = []
    for neg_class in neg_classes:
        neg_features.extend(feature_vectors[neg_class])
    
    if equal_features:
        neg_features = np.random.permutation(neg_features)[:pos_features.shape[0]]
    
    if is_train:
        return pos_features[:TRAIN_SPLIT], np.array(neg_features[:TRAIN_SPLIT])
    else:
        return pos_features[TRAIN_SPLIT:], np.array(neg_features[TRAIN_SPLIT:])
    
def compute_accuracy(weight_vector, pos_features, neg_features):
    classifier = pegasos.PegasosSVMClassifier()
    classifier.fit(np.zeros((2, 1024)), np.asarray([1, 0]))
    classifier.weight_vector.weights = weight_vector

    # Concat data and pass to SVM
    result = classifier.predict(np.vstack((pos_features, neg_features)))
    ground_truth = np.concatenate((np.ones(len(pos_features)), np.zeros(len(neg_features))))
    return np.average(np.equal(ground_truth, result))

def get_svm_weights(x_train, y_train):
    svm = pegasos.PegasosSVMClassifier()
    svm.fit(x_train, y_train)
    weight_vector = svm.weight_vector.weights
    return weight_vector


def get_x_y(pos_features, neg_features):
    x = np.vstack((pos_features, neg_features))
    y = np.hstack((np.ones( len(pos_features)),
                   np.zeros(len(neg_features))))
    return x, y

def compute_X1(pos_class, base_classes, feature_vectors):
    neg_classes = np.delete(base_classes, np.argwhere(base_classes==pos_class))
    pos_features, neg_features = prepare_features(pos_class, neg_classes, feature_vectors)
    x_train, y_train = get_x_y(pos_features, neg_features)
    weight_vector = get_svm_weights(x_train, y_train)
    return weight_vector

def compute_X2(pos_class, base_classes, feature_vectors):
    pos_features, neg_features = prepare_features(pos_class, base_classes, feature_vectors)
    x_train, y_train = get_x_y(pos_features, neg_features)
    weight_vector = get_svm_weights(x_train, y_train)
    return weight_vector
    
def compute_Y(pos_class, new_class, base_classes, feature_vectors):
    neg_classes = np.delete(base_classes, np.argwhere(base_classes==pos_class))
    neg_classes = np.append(neg_classes, new_class)
    pos_features, neg_features = prepare_features(pos_class, neg_classes, feature_vectors)
    x_train, y_train = get_x_y(pos_features, neg_features)
    weight_vector = get_svm_weights(x_train, y_train)
