import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from dataclasses import dataclass, field

from imblearn.over_sampling import SMOTE
from scipy import interp

from multiprocessing import Queue, Process, freeze_support
from BCa_bootstrap import Est_bca_CI
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    auc,
    roc_curve,
)

from itertools import product
import matplotlib.pyplot as plt

# Function to assign binary classification rule based on cutoff
# and a predetermined rule (lower or greater than cutoff)


def binary_rule(score: pd.Series, cutoff: float, rule: str) -> pd.Series:
    """
    Function to assign binary classification rule based on cutoff
    and a predetermined rule (lower or greater than cutoff)

    Parameters
    ----------
    score : pd.Series
        A pandas series containing the scores.
    cutoff : float
        The cutoff value.
    rule : str
        The rule to use. Either 'lower' or 'greater'.

    Returns
    -------
    pd.Series
        A pandas series containing the binary classification rule.
    """

    if rule == "lt":
        return score.apply(lambda x: 1 if x < cutoff else 0)
    elif rule == "gt":
        return score.apply(lambda x: 1 if x > cutoff else 0)
    else:
        raise ValueError("Rule must be either 'lt' or 'gt'.")


def plot_confusion_matrix(
    cm: np.ndarray, classes: List[int], normalize: bool, title: str, cmap: plt.cm
) -> None:

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(6, 6))

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".3f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("Truth")
    plt.xlabel("Classified")
    plt.show()


def make_confusion_mat(
    truth: pd.Series, pred: pd.Series, normalize: bool
) -> np.ndarray:

    """Function to make a confusion matrix

    Parameters
    ----------
    truth : pd.Series
        Truth data (binary values)
    pred : pd.Series
        Prediction data (binary values)
    normalize : bool
        Whether to normalize the confusion matrix or not.
    Returns
    -------
    np.ndarray
        A confusion matrix.
    """
    # Create a 2x2 confusion matrix
    cm = confusion_matrix(truth, pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return cm


def diagnostic_perfm_metrics(truth: pd.Series, pred: pd.Series) -> Dict:

    """
    Function to estimate the performance metrics of a diagnostic test
    from truth and prediction data, and a critical cutoff value.
    including: sensitivity, specificity, balanced accuracy, F1-score,
    FPR, FNR, PPV, NPV, LR+, LR-.

    Parameters
    ----------
    truth : pd.Series
        Truth data (binary values)
    pred : pd.Series
        classified data (binary values)
    Returns
    -------
    Dict
        A dictionary containing the performance metrics.
    """

    # Create holder for metrics
    metrics = {}

    # Create a 2x2 confusion matrix
    cm = confusion_matrix(truth, pred)

    TN, FP, FN, TP = cm.ravel()
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)

    SEN = TPR
    SPEC = TNR
    ACC = (TP + TN) / (TP + FP + TN + FN)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)

    # Estimate LR+ and LR-
    if SPEC == 1:
        LRP = np.inf
    else:
        LRP = SEN / (1 - SPEC)

    if SPEC == 0:
        LRN = np.inf
    else:
        LRN = (1 - SEN) / SPEC

    # Estimate F1-score and BAC
    F1 = f1_score(truth, pred)
    BAC = balanced_accuracy_score(truth, pred)

    # Youden J statistic
    YJ = TPR + TNR - 1

    # Add metrics to dictionary
    metrics["Sens"] = SEN
    metrics["Spec"] = SPEC
    metrics["FPR"] = FPR
    metrics["FNR"] = FNR
    metrics["PPV"] = PPV
    metrics["NPV"] = NPV
    metrics["LRp"] = LRP
    metrics["LRn"] = LRN
    metrics["F1"] = F1
    metrics["BAC"] = BAC
    metrics["Youden_J"] = YJ

    return metrics


# Function to estimate BCA CI for the above metrics with multithreading
def Bootstrap_Diag_perfm(
    truth: pd.Series,
    pred: pd.Series,
    alpha: float,
    n_samples: int,
    n_jobs: int,
) -> Dict:

    """Function to estimate the BCa-CI for performance metrics for a binary classification rule
    based on a clinical score
    including: sensitivity, specificity, balanced accuracy, F1-score,
    FPR, FNR, PPV, NPV, LR+, LR-

    Parameters
    ----------
    truth : pd.Series
        Truth data (binary values)
    pred : pd.Series
        classified data (binary values)
    alpha: float
        Significance level for the BCa-CI.
    n_samples: int
        Number of samples to draw from the original dataset.
    n_jobs : int
        Number of processes to run in parallel.
    Returns
    -------
    Dict
        A dictionary containing the performance metrics and their BCa-CI.
    """

    # Results holder
    res = locals()

    # Estimate empirical values of metrics
    emp_metrics = diagnostic_perfm_metrics(truth, pred)

    res["emp_metrics"] = emp_metrics

    collection = Queue()
    thread_list = []

    # A dictionary holder for coolecting bootstrap values for metrics as np.array
    b_holder = {}

    for k in emp_metrics.keys():
        b_holder[k] = np.array([])

    # Establishing the bootstrap process
    with tqdm(total=n_jobs) as pbar:

        for _ in range(n_jobs):
            p = Process(target=diag_perfm_worker, args=(res, collection))
            p.start()
            thread_list.append(p)

        # Collecting the results from collection and append each one
        # to the corresponding np.array in b_holder

        for _ in range(n_jobs):
            b_output = collection.get()
            for k in b_holder.keys():
                b_holder[k] = np.append(b_holder[k], b_output[k])
            pbar.update(1)

        for p in thread_list:
            p.join()

    # Create a report as dictionary containing only BCa_CIs for each metric
    report = {}

    # Estimate the BCa-CI for each metric if possible,
    # otherwise determine CI as percentiles corresponding to alpha
    for k in b_holder.keys():
        emp_v = emp_metrics[k]
        try:
            report[k + "_CI"] = Est_bca_CI(
                boot_thetas=b_holder[k], emp_val=emp_v, alpha=alpha
            )
        except:
            report[k + "_CI"] = np.percentile(b_holder[k], [alpha / 2, 1 - alpha / 2])

    return emp_metrics, report


def diag_perfm_worker(res: Dict, collection: Queue) -> None:

    """Worker function for the Bootstrap_Diag_perfm function.

    Parameters
    ----------
    res : Dict
        A dictionary containing the local data and params.

    collection : Queue
        A queue to collect the results.

    Returns
    -------
    None
    """

    # A dictionary holder for coolecting bootstrap values for metrics as np.array
    outputs = {}

    for k in res["emp_metrics"].keys():
        outputs[k] = np.array([])

    reps = (res["n_samples"] // res["n_jobs"]) + 1

    for _ in range(reps):

        # Bootstrap sampling
        b_truth, b_pred = diag_perf_bootstrap_sample(res["truth"], res["pred"])

        # Estimate the metrics for the bootstrap sample
        b_metrics = diagnostic_perfm_metrics(b_truth, b_pred)

        # Save the metrics to outputs
        for k, v in b_metrics.items():
            outputs[k] = np.append(outputs[k], v)

    # Add the results to the collection
    collection.put(outputs)


# Function to create bootstrap samples
def diag_perf_bootstrap_sample(
    truth: pd.Series, pred: pd.Series
) -> Tuple[pd.Series, pd.Series]:

    """Function to create bootstrap samples

    Parameters
    ----------
    truth : pd.Series
        Series of truth values

    pred : pd.Series
        Series of predicted values

    Note: truth and pred are paired

    Returns
    -------
    b_truth : np.ndarray
        Array of bootstrap samples of truth values

    b_pred : np.ndarray
        Array of bootstrap samples of predicted values
    """

    # Joining truth and pred into a dataframe
    data = pd.concat([truth, pred], axis=1)

    # Creating bootstrap samples on data
    b_data = data.sample(frac=1, replace=True)

    # Splitting the data into truth and pred
    b_truth = b_data.iloc[:, 0]
    b_pred = b_data.iloc[:, 1]

    return b_truth, b_pred


# Class to establish the best cutoff value for a binary classification
# problem based on a given score, truth and predetermined rule.


@dataclass
class Best_cutoff:

    score: pd.Series = field(init=True)
    truth: pd.Series
    rule: str = field(init=True)
    balance: bool = field(init=True)

    def _post_init(self):

        if self.balance:
            # initialize a SMOTE oversampling objec
            self.sm = SMOTE(random_state=123)
            X, Y = self.sm.fit_resample(self.score.values.reshape(-1, 1), self.truth)
            self.score = pd.Series(X.ravel())
            self.truth = pd.Series(Y)

    def optimize_cut(
        self, resolution: float, xmin: float, xmax: float, metric: str, crit: str
    ) -> pd.DataFrame:

        """Function to optimize the cutoff value for a binary classification problem.

        Parameters
        ----------
        resolution : float
            The resolution of the score scale
        xmin : float
            The minimum value of the score scale
        xmax : float
            The maximum value of the score scale
        crit : str
            The criterion to optimize the cutoff value. It can be either 'max' or 'min'.
        metric : str
            The metric to optimize the cutoff value.
            supporting conventional metrics for binary classification problems.

        Returns
        -------
        dataframe with the following columns:
            'Cutoff', 'Sens', 'Spec', 'F1', 'BAC', 'PPV', 'NPV',
            'LRp', 'LRn', 'FPR', 'FNR', 'Youden_J'
        """

        # Create a list of all possible cutoff values
        thres_list = np.linspace(xmin, xmax, int((xmax - xmin) / resolution))

        # Create Holder for the performance metrics as a pd.Dataframe with the first column indicates the cutoff values
        temp_perf = pd.DataFrame(
            columns=[
                "Cutoff",
                "Sens",
                "Spec",
                "F1",
                "BAC",
                "PPV",
                "NPV",
                "LRp",
                "LRn",
                "FPR",
                "FNR",
                "Youden_J",
            ]
        )

        i = 0
        for t in thres_list:

            score, pred = self.score, binary_rule(self.score, t, self.rule)

            # Calculate the performance metrics using diagnostic_perfm_metrics function
            perf = diagnostic_perfm_metrics(truth=self.truth, pred=pred)
            i += 1

            # Transform the performance metrics into pd.Dataframe
            perf = pd.DataFrame(perf, index=[i])
            # Add the cutoff value to the performance metrics as the first column
            perf.insert(0, "Cutoff", t)

            # Append the performance metrics to temp_perf
            temp_perf = temp_perf.append(perf, ignore_index=True)

        self.temp_perf = temp_perf

        # Select the cutoff value that maximizes the given criterion
        print(f"Xác định ngưỡng cắt tối ưu theo tiêu chí {crit} {metric}")

        if crit == "max":
            best_cut = temp_perf.loc[temp_perf[metric].idxmax(), "Cutoff"]
        elif crit == "min":
            best_cut = temp_perf.loc[temp_perf[metric].idxmin(), "Cutoff"]

        best_perfm = temp_perf.loc[temp_perf["Cutoff"] == best_cut]

        self.best_cut = best_cut
        self.perf_df = temp_perf
        self.best_perfm = best_perfm
        self.xmin = xmin
        self.xmax = xmax

        # Draw the calibration plot
        self.calibration_plot()

        return best_perfm

    def calibration_plot(self):

        temp_perf = self.temp_perf

        plt.rcParams.update({"font.size": 10})
        plt.figure(figsize=(10, 5))

        plt.plot(temp_perf["Cutoff"], temp_perf["Sens"], label="Độ nhạy", linewidth=1.5)
        plt.plot(
            temp_perf["Cutoff"], temp_perf["Spec"], label="Độ đặc hiệu", linewidth=1.5
        )
        plt.plot(temp_perf["Cutoff"], temp_perf["F1"], label="F1", linewidth=1.5)
        plt.plot(
            temp_perf["Cutoff"], temp_perf["BAC"], label="Độ chính xác", linewidth=1.5
        )
        plt.plot(
            temp_perf["Cutoff"], temp_perf["Youden_J"], label="Youden_J", linewidth=1.5
        )

        bc = self.best_cut
        xmin, xmax = self.xmin, self.xmax

        plt.vlines(
            x=bc,
            ymin=0.1,
            ymax=1,
            colors="k",
            linestyles="dashed",
            label=f"Ngưỡng tối ưu:{bc:.2f}",
        )

        plt.xlabel("Thang đo", fontsize=15)
        plt.ylabel("Điểm số", fontsize=15)
        plt.legend(fontsize=12)
        plt.ylim([0.1, 1])
        plt.xlim([xmin, xmax])

        plt.show()

    # function to draw the ROC curve with Bootstrap CI

    def roc_curve(self, n_iter=1000, rule: str = "lt"):

        score, pred, truth = (
            self.score,
            binary_rule(self.score, self.best_cut, self.rule),
            self.truth,
        )

        # Holders for tprs and aucs
        tprs = []
        aucs = []

        base_fpr = np.linspace(0, 1, 101)

        valid_df = pd.DataFrame(
            {"Score": score, "Pred": pred, "Truth": truth}, index=score.index
        )

        for i in range(n_iter):
            resamp_df = valid_df.sample(1000, replace=True)

            if len(np.unique(resamp_df["Pred"])) < 2:
                continue

            if rule:
                fpr, tpr, _ = roc_curve(
                    resamp_df["Truth"], resamp_df["Score"], pos_label=0
                )
            else:
                fpr, tpr, _ = roc_curve(
                    resamp_df["Truth"], resamp_df["Score"], pos_label=1
                )

            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        mean_auc = auc(base_fpr, mean_tprs)
        std_auc = np.std(aucs)

        tprs_up = np.minimum(mean_tprs + 1.645 * std, 1)
        tprs_low = mean_tprs - 1.645 * std

        plt.rcParams["figure.figsize"] = (7, 6.6)
        plt.rcParams.update({"font.size": 12})

        plt.plot(
            base_fpr,
            mean_tprs,
            "r",
            alpha=1,
            label=r"AUC = %0.3f $\pm$ %0.3f" % (mean_auc, std_auc),
        )

        plt.fill_between(base_fpr, tprs_low, tprs_up, color="red", alpha=0.3)

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=2,
            color="grey",
            label="Đoán ngẫu nhiên",
            alpha=0.5,
        )

        plt.vlines(
            x=1 - self.best_perfm["Spec"],
            ymin=0.0,
            ymax=1,
            colors="k",
            linewidth=0.8,
            linestyles="dashed",
        )

        plt.hlines(
            y=self.best_perfm["Sens"],
            xmin=0.0,
            xmax=1,
            colors="k",
            linewidth=0.8,
            linestyles="dashed",
        )

        plt.plot(1 - self.best_perfm["Spec"], self.best_perfm["Sens"], "ko")

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.ylabel("Tỉ lệ dương tính thật")
        plt.xlabel("Tỉ lệ dương tính giả")
        plt.legend(loc="lower right")

        plt.show()