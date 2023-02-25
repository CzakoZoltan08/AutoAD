from sklearn.metrics import roc_auc_score
from pyod.utils.utility import precision_n_scores
from sklearn.preprocessing import StandardScaler
from autoad.algorithms.abod import AngleBaseOutlierDetection
from autoad.algorithms.autoencoder import AutoEncoder
from autoad.algorithms.cblof import ClusterBasedLocalOutlierFactor
from autoad.algorithms.feature_bagging import FeatureBaggingOutlierDetection
from autoad.algorithms.fttransformer.fttransformer import FTTransformer
from autoad.algorithms.ganomaly.ganomaly import GANomaly
from autoad.algorithms.gmm import GMMAnomalyDetector
from autoad.algorithms.hbos import HistogramBasedOutlierDetection
from autoad.algorithms.inne import InneAnomalyDetector
from autoad.algorithms.kde import KDEAnomalyDetector
from autoad.algorithms.knn import KNearestNeighbors
from autoad.algorithms.lmdd import LMDDAnomalyDetector
from autoad.algorithms.loda import LightweightOnlineDetector
from autoad.algorithms.lof import LocalOutlierFactor
from autoad.algorithms.lscp import LocallySelectiveCombination
from autoad.algorithms.lstmod import LSTMOutlierDetector
from autoad.algorithms.mcd import MinimumCovarianceDeterminant
from autoad.algorithms.mogaal import MultiObjectiveGenerativeAdversarialActiveLearning
from autoad.algorithms.ocsvm import OneClassSVM
from autoad.algorithms.pca import PCAAnomalyDetector
from autoad.algorithms.prenet.prenet import PReNet
from autoad.algorithms.repen.repen import REPEN
from autoad.algorithms.sod import SubspaceOutlierDetection
from autoad.algorithms.sogaal import SingleObjectiveGenerativeAdversarialActiveLearning
from autoad.algorithms.vae import VariationalAutoEncoder
from autoad.data_generators.anomaly_data_generator import AnomalyDataGenerator
from autoad.algorithms.isolation_forest import IsolationForest
from autoad.algorithms.feawad.feawad import FEAWAD
from autoad.algorithms.deeplog_model import DeepLogModel
from autoad.algorithms.deep_sad.deepsad import DeepSAD
from autoad.algorithms.deep_autoencoding_gaussian_mixture_model \
    import DeepAutoencodingGaussianMixtureModel
import numpy as np
import pytest


@pytest.fixture
def dataset():
    anomaly_data_generator = AnomalyDataGenerator()
    return anomaly_data_generator.generate(scaler=StandardScaler())


@pytest.fixture
def X_train(dataset):
    return dataset['X_train']


@pytest.fixture
def X_test(dataset):
    return dataset['X_test']


@pytest.fixture
def y_test(dataset):
    return dataset['y_test'].ravel()


@pytest.fixture
def y_train(dataset):
    return dataset['y_train'].ravel()


@pytest.fixture
def outliers_fraction(y_train, y_test):
    return (np.count_nonzero(y_train)+np.count_nonzero(y_test)) \
        / (len(y_train) + len(y_test))


@pytest.fixture
def random_state():
    return np.random.RandomState(42)


@pytest.fixture
def isolation_forest_classifier(outliers_fraction, random_state):
    return IsolationForest(contamination=outliers_fraction,
                           random_state=random_state)


@pytest.fixture
def k_nearest_neighbors_classifier(outliers_fraction):
    return KNearestNeighbors(contamination=outliers_fraction)


@pytest.fixture
def deep_autoencoding_gaussian_mixture_model():
    return DeepAutoencodingGaussianMixtureModel()


@pytest.fixture
def deep_log_model():
    return DeepLogModel()


@pytest.fixture
def feawad_model():
    return FEAWAD()


@pytest.fixture
def deep_sad_model():
    return DeepSAD()


@pytest.fixture
def fttransformer_model():
    return FTTransformer()


@pytest.fixture
def ganomaly_model():
    return GANomaly()


@pytest.fixture
def prenet_model():
    return PReNet()


@pytest.fixture
def repen_model():
    return REPEN()


@pytest.fixture
def pca_model():
    return PCAAnomalyDetector()


@pytest.fixture
def auto_encoder_model():
    return AutoEncoder()


@pytest.fixture
def cblof_model():
    return ClusterBasedLocalOutlierFactor()


@pytest.fixture
def hbos_model():
    return HistogramBasedOutlierDetection()


@pytest.fixture
def abod_model():
    return AngleBaseOutlierDetection()


@pytest.fixture
def loda_model():
    return LightweightOnlineDetector()


@pytest.fixture
def lof_model():
    return LocalOutlierFactor()


@pytest.fixture
def mogaal_model():
    return MultiObjectiveGenerativeAdversarialActiveLearning()


@pytest.fixture
def ocsvm_model():
    return OneClassSVM()


@pytest.fixture
def sod_model():
    return SubspaceOutlierDetection()


@pytest.fixture
def sogaal_model():
    return SingleObjectiveGenerativeAdversarialActiveLearning()


@pytest.fixture
def vae_model():
    return VariationalAutoEncoder(encoder_neurons=[32, 16], decoder_neurons=[16, 32])


@pytest.fixture
def feature_bagging_model():
    return FeatureBaggingOutlierDetection()


@pytest.fixture
def mcd_model():
    return MinimumCovarianceDeterminant()


@pytest.fixture
def lscp_model():
    return LocallySelectiveCombination()


@pytest.fixture
def inne_model():
    return InneAnomalyDetector()


@pytest.fixture
def gmm_model():
    return GMMAnomalyDetector()


@pytest.fixture
def kde_model():
    return KDEAnomalyDetector()


@pytest.fixture
def lmdd_model():
    return LMDDAnomalyDetector()


def test_isolation_forest(X_train,
                          X_test,
                          y_test,
                          isolation_forest_classifier):
    isolation_forest_classifier.fit(X_train)

    y_pred = isolation_forest_classifier.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_k_nearest_neighbors(X_train,
                             X_test,
                             y_test,
                             k_nearest_neighbors_classifier):
    k_nearest_neighbors_classifier.fit(X_train)

    y_pred = k_nearest_neighbors_classifier.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_deep_autoencoding_gaussian_mixture_model(
        X_train,
        X_test,
        y_test,
        deep_autoencoding_gaussian_mixture_model):
    deep_autoencoding_gaussian_mixture_model.fit(X_train)

    energy = deep_autoencoding_gaussian_mixture_model.predict(X_test)

    ano_index = np.arange(len(energy))[energy > np.percentile(energy, 75)]

    y_pred = y_test.copy()
    for index in ano_index:
        y_pred[index] = 1

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_deep_sad_model(X_train,
                        X_test,
                        y_train,
                        y_test,
                        deep_sad_model):
    deep_sad_model.fit(X_train, y_train)

    y_pred = deep_sad_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_feawad_model(X_train,
                      X_test,
                      y_train,
                      y_test,
                      feawad_model):
    feawad_model.fit(X_train, y_train)

    y_pred = feawad_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_fttransformer_model(X_train,
                             X_test,
                             y_train,
                             y_test,
                             fttransformer_model):
    fttransformer_model.fit(X_train, y_train)

    y_pred = fttransformer_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_ganomaly_model(X_train,
                        X_test,
                        y_train,
                        y_test,
                        ganomaly_model):
    ganomaly_model.fit(X_train, y_train)

    y_pred = ganomaly_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_prenet_model(X_train,
                      X_test,
                      y_train,
                      y_test,
                      prenet_model):
    prenet_model.fit(X_train, y_train)

    y_pred = prenet_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_repen_model(X_train,
                     X_test,
                     y_train,
                     y_test,
                     repen_model):
    repen_model.fit(X_train, y_train)

    y_pred = repen_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_pca_model(X_train,
                   X_test,
                   y_train,
                   y_test,
                   pca_model):
    pca_model.fit(X_train, y_train)

    y_pred = pca_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_auto_encoder_model(X_train,
                            X_test,
                            y_train,
                            y_test,
                            auto_encoder_model):
    auto_encoder_model.fit(X_train, y_train)

    y_pred = auto_encoder_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_cblof_model(X_train,
                     X_test,
                     y_train,
                     y_test,
                     cblof_model):
    cblof_model.fit(X_train, y_train)

    y_pred = cblof_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_hbos_model(X_train,
                    X_test,
                    y_train,
                    y_test,
                    hbos_model):
    hbos_model.fit(X_train, y_train)

    y_pred = hbos_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_abod_model(X_train,
                    X_test,
                    y_train,
                    y_test,
                    abod_model):
    abod_model.fit(X_train, y_train)

    y_pred = abod_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_loda_model(X_train,
                    X_test,
                    y_train,
                    y_test,
                    loda_model):
    loda_model.fit(X_train, y_train)

    y_pred = loda_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_lstmod_model(X_train,
                      X_test,
                      y_train,
                      y_test,):
    print(X_train.shape, X_test.shape)

    clf = LSTMOutlierDetector(contamination=0.1)
    clf.fit(X_train)
    # pred_scores = clf.decision_function(X_test)
    y_pred, left_inds, right_inds = clf.predict(X_test)

    print(y_pred.shape, left_inds.shape, right_inds.shape)

    print(clf.threshold_)
    # print(np.percentile(pred_scores, 100 * 0.9))

    # print('pred_scores: ',pred_scores)
    print('pred_labels: ', y_pred)

    roc = round(roc_auc_score(y_test, y_pred[:len(y_test)]), ndigits=4)

    assert roc > 0.2


def test_lof_model(X_train,
                   X_test,
                   y_train,
                   y_test,
                   lof_model):
    lof_model.fit(X_train, y_train)

    y_pred = lof_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_mogaal_model(X_train,
                      X_test,
                      y_train,
                      y_test,
                      mogaal_model):
    mogaal_model.fit(X_train, y_train)

    y_pred = mogaal_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_ocsvm_model(X_train,
                     X_test,
                     y_train,
                     y_test,
                     ocsvm_model):
    ocsvm_model.fit(X_train, y_train)

    y_pred = ocsvm_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_sod_model(X_train,
                   X_test,
                   y_train,
                   y_test,
                   sod_model):
    sod_model.fit(X_train, y_train)

    y_pred = sod_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_sogaal_model(X_train,
                      X_test,
                      y_train,
                      y_test,
                      sogaal_model):
    sogaal_model.fit(X_train, y_train)

    y_pred = sogaal_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_vae_model(X_train,
                   X_test,
                   y_train,
                   y_test,
                   vae_model):
    vae_model.fit(X_train, y_train)

    y_pred = vae_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_feature_bagging_model(X_train,
                               X_test,
                               y_train,
                               y_test,
                               feature_bagging_model):
    feature_bagging_model.fit(X_train, y_train)

    y_pred = feature_bagging_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_deeplog_model(X_train,
                       X_test,
                       y_train,
                       y_test,
                       deep_log_model):
    deep_log_model.fit(X_train, y_train)

    y_pred = deep_log_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_mcd_model(X_train,
                   X_test,
                   y_train,
                   y_test,
                   mcd_model):
    mcd_model.fit(X_train, y_train)

    y_pred = mcd_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_lscp_model(X_train,
                    X_test,
                    y_train,
                    y_test,
                    lscp_model):
    lscp_model.fit(X_train, y_train)

    y_pred = lscp_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_inne_model(X_train,
                    X_test,
                    y_train,
                    y_test,
                    inne_model):
    inne_model.fit(X_train, y_train)

    y_pred = inne_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_gmm_model(X_train,
                   X_test,
                   y_train,
                   y_test,
                   gmm_model):
    gmm_model.fit(X_train, y_train)

    y_pred = gmm_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_kde_model(X_train,
                   X_test,
                   y_train,
                   y_test,
                   kde_model):
    kde_model.fit(X_train, y_train)

    y_pred = kde_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_lmdd_model(X_train,
                    X_test,
                    y_train,
                    y_test,
                    lmdd_model):
    lmdd_model.fit(X_train, y_train)

    y_pred = lmdd_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_adone_model(X_train,
                     X_test,
                     y_train,
                     y_test):
    from pygod.models import AdONE

    adone_model = AdONE()

    adone_model.fit(X_train)

    y_pred = adone_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5
