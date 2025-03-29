import sys
import numpy as np
import pandas as pd
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy
from IPython.display import display
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult
from clearbox_engine import Dataset, Preprocessor, TabularEngine, LabeledSynthesizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#bias mitigation algorithms
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.disparate_impact_remover import DisparateImpactRemover
#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#datasets
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult
from aif360.datasets import AdultDataset
from aif360.datasets import MEPSDataset19
from aif360.datasets import StandardDataset



# Ensure reproducibility
np.random.seed(1)

# Append a path if needed
sys.path.append("../")

def load_and_process_data(dataset_name, use_disparate_impact_remover=False):
    """
    Load and preprocess the dataset based on the name provided.
    Optionally, use DisparateImpactRemover based on the flag.
    """
    if dataset_name == 'adult':
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        
        if use_disparate_impact_remover:
            # Use DisparateImpactRemover approach
            preprocessed_dataset = AdultDataset(
                protected_attribute_names=['sex'],
                privileged_classes=[['Male']],
                categorical_features=[],
                features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            )
        else:
            # Use the preprocessed dataset for ADULT with bias mitigation algorithms: Reweighing, LFR, OptimPreproc
            preprocessed_dataset = load_preproc_data_adult(['sex'])
        
        # Splitting the dataset
        train, val_test = preprocessed_dataset.split([0.7], shuffle=True, seed=42)
        val, test = val_test.split([0.5], shuffle=True, seed=42)
        
        return train, val, test, privileged_groups, unprivileged_groups
        

    elif dataset_name == 'meps':
        def preprocess_meps_dataset(meps_dataset, use_disparate_impact_remover=False):
            """
            Preprocess the MEPS dataset for AIF360, with the option to apply 
            DisparateImpactRemover and removing categorical features.
            """
            # Convert MEPSDataset19 to a DataFrame
            df, metadata = meps_dataset.convert_to_dataframe()

            # Clean invalid values for PCS42 and MCS42
            for col in ['PCS42', 'MCS42']:
                if col in df.columns:
                    # Replace negative values with NaN
                    df[col] = df[col].apply(lambda x: pd.NA if x < 0 else x)
                    # Fill NaN values with the column median
                    df[col] = df[col].fillna(df[col].median(skipna=True))

            df.rename(columns={'SEX=1': 'SEX'}, inplace=True)
            df['RACE'] = df['RACE'].replace({'White': 1.0, 'Non-White': 0.0})
            
            # Step 2: Handle scaling and feature renaming
            if use_disparate_impact_remover:
                # For DisparateImpactRemover: Remove categorical features
                # Retain only numerical features
                selected_columns = ['RACE', 'SEX', 'PCS42', 'MCS42', 'UTILIZATION']
                df = df[selected_columns]
            else:
                # For regular preprocessing: Retain categorical features (e.g., age groups)
                # Group age into decades
                df['Age (decade)'] = df['AGE'].apply(lambda x: min(x // 10 * 10, 70))
                # One-hot encode categorical columns
                categorical_columns = ['Age (decade)']
                df = pd.get_dummies(df, columns=categorical_columns)
                # Rename one-hot encoded columns to remove `.0` suffix
                df.rename(columns=lambda col: col.replace('.0', '') if 'Age (decade)' in col else col, inplace=True)
                # Dynamically retrieve column names for encoded categories
                age_decade_columns = [col for col in df.columns if 'Age (decade)_' in col]
                # Include new features (POVCAT and INSCOV)
                additional_features = [
                    'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                    'INSCOV=1', 'INSCOV=2', 'INSCOV=3'
                ]
                #Retain only necessary columns
                selected_columns = (
                    ['RACE', 'SEX', 'PCS42', 'MCS42'] +  # Include numerical and protected attributes
                    age_decade_columns +
                    additional_features +
                    ['UTILIZATION']  # Include target variable
                )
                df = df[selected_columns]

            # Create the processed AIF360 dataset
            processed_dataset = StandardDataset(
                df,
                label_name='UTILIZATION',
                favorable_classes=[1.0],
                protected_attribute_names=['RACE', 'SEX'],
                privileged_classes=[[1.0], [1.0]],  # Privileged groups: White and Male
            )

            return processed_dataset
                
   
        privileged_groups = [{'RACE': 1}]
        unprivileged_groups = [{'RACE': 0}]
        
        # Use the appropriate preprocessing based on whether DisparateImpactRemover is needed
        meps = MEPSDataset19()
        processed_meps = preprocess_meps_dataset(meps, use_disparate_impact_remover)

        # Splitting the MEPS dataset
        train, val_test = processed_meps.split([0.7], shuffle=True, seed=42)
        val, test = val_test.split([0.5], shuffle=True, seed=42)
        
        return train, val, test, privileged_groups, unprivileged_groups
            

def apply_bias_mitigation(method, train, test, unprivileged_groups, privileged_groups):
    
    if method == 'reweighing':
        reweighing = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
        reweighing.fit(train)
        train_transf = reweighing.transform(train)
        test_transf = reweighing.transform(test)
    
    elif method == 'lfr':
        lfr = LFR(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
                k=10, Ax=0.1, Ay=1.0, Az=1.5,
                verbose=1)
        train_transf = lfr.fit_transform(train)
        test_transf = lfr.transform(test)
    
    elif method == 'optimpreproc':
        optim_options = {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        } 

        optim_preproc = OptimPreproc(OptTools, optim_options,
                        unprivileged_groups = unprivileged_groups,
                        privileged_groups = privileged_groups)

        optim_preproc = optim_preproc.fit(train)
        train_transf = optim_preproc.transform(train)
        train_transf = train.align_datasets(train_transf)

        test_transf = optim_preproc.transform(test)
        test_transf = test.align_datasets(test_transf)
    
    elif method == 'disparateimpactremover':
        disp_impact_remover = DisparateImpactRemover(repair_level= 1.0, sensitive_attribute="sex")
        train_transf = disp_impact_remover.fit_transform(train)
        test_transf = disp_impact_remover.fit_transform(test)
    
    elif method == 'synthetic':
        # to be done later
        pass
    else:
        raise ValueError("Unsupported bias mitigation method. Choose from 'reweighing', 'lfr', 'optimpreproc', 'disparateimpactremover', 'synthetic'.")

    return train_transf, test_transf

def evaluate_fairness_metrics(dataset, unprivileged_groups, privileged_groups, label="Dataset"):
    metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups, privileged_groups)
    
    stat_parity_diff = f"{metric.mean_difference():.4f}"
    disp_impact = f"{metric.disparate_impact():.4f}"
    
    return stat_parity_diff, disp_impact


def train_classifier_and_find_best_threshold(train, val, classifier_type, unprivileged_groups, privileged_groups):

    
    # Select classifier
    if classifier_type == 'logistic_regression':
        #classifier = LogisticRegression(random_state=1, solver='liblinear', max_iter=1000)   #depends what logstic regression u use here
        classifier = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=1)
        #classifier = LogisticRegression(random_state=1)


    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=1)
    elif classifier_type == 'xgboost':
        classifier = XGBClassifier(random_state=1)
    else:
        raise ValueError("Invalid classifier type specified.")
    
    # Train the classifier on the training data
    classifier.fit(train.features, train.labels.ravel(), sample_weight=train.instance_weights)
    
    # Predict on the validation set
    val_scores = classifier.predict_proba(val.features)[:, 1]  # Only for the favorable class

    # Select best threshold for validation set
    thresholds = np.arange(0.01, 1, 0.01)
    val_balanced_accs = []

    for threshold in thresholds:
        val_LR_predictions = (val_scores >= threshold).astype(int)

        val_with_LR_scores = copy.deepcopy(val)
        val_with_LR_scores.labels = val_LR_predictions.reshape(-1, 1)  # Update labels with predictions

        val_metric = ClassificationMetric(val, val_with_LR_scores,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        balanced_acc = (val_metric.true_positive_rate() + val_metric.true_negative_rate()) / 2

        val_balanced_accs.append(balanced_acc)

    # Find the best threshold and associated balanced accuracy
    best_index = np.argmax(val_balanced_accs)
    best_threshold = thresholds[best_index]
    best_balanced_acc = val_balanced_accs[best_index]

    print(f"Best Threshold: {best_threshold}")
    print(f"Best Balanced Accuracy: {best_balanced_acc}")

    return classifier, best_threshold, best_balanced_acc

def apply_best_threshold_and_compute_metrics(classifier, test, best_threshold, unprivileged_groups, privileged_groups):
    # Apply the best threshold to the test set
    test_scores = classifier.predict_proba(test.features)[:, 1]
    test_LR_predictions= (test_scores >= best_threshold).astype(int)

    # Create a copy of the test dataset and set predicted labels
    test_with_LR_scores = copy.deepcopy(test)
    test_with_LR_scores.labels = test_LR_predictions.reshape(-1, 1)

    # Calculate fairness and performance metrics on the test set
    test_metric = ClassificationMetric(test, test_with_LR_scores,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)
    # Collect the metrics in a dictionary
    metrics = {
        'balanced_accuracy': (test_metric.true_positive_rate() + test_metric.true_negative_rate()) / 2,
        'statistical_parity_difference': test_metric.statistical_parity_difference(),
        'disparate_impact': test_metric.disparate_impact(),
        'average_odds_difference': test_metric.average_odds_difference(),
        'equal_opportunity_difference': test_metric.equal_opportunity_difference(),
        'theil_index': test_metric.theil_index()
    }

    # Print the metrics
    print(f"Balanced Accuracy (test): {metrics['balanced_accuracy']}")
    print(f"Statistical Parity Difference (test): {metrics['statistical_parity_difference']}")
    print(f"Disparate Impact (test): {metrics['disparate_impact']}")
    print(f"Average Odds Difference (test): {metrics['average_odds_difference']}")
    print(f"Equal Opportunity Difference (test): {metrics['equal_opportunity_difference']}")
    print(f"Theil Index (test): {metrics['theil_index']}")

    return metrics, test_scores

def plot_metrics(test, test_scores, best_threshold, unprivileged_groups, privileged_groups, thresholds=np.arange(0.01, 1, 0.01)):

    test_balanced_accs = []
    test_disp_impacts = []
    test_avg_odds_diffs = []

    for threshold in thresholds:
        test_predictions = (test_scores >= threshold).astype(int)
        test_with_scores = copy.deepcopy(test)
        test_with_scores.labels = test_predictions.reshape(-1, 1)

        # Compute fairness and performance metrics
        test_metric = ClassificationMetric(test, test_with_scores, unprivileged_groups, privileged_groups)
        
        test_balanced_accs.append((test_metric.true_positive_rate() + test_metric.true_negative_rate()) / 2)
        test_disp_impacts.append(test_metric.disparate_impact())
        test_avg_odds_diffs.append(test_metric.average_odds_difference())

    # Plot Balanced Accuracy and Fairness Metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Balanced Accuracy
    ax1.plot(thresholds, test_balanced_accs, label="Balanced Accuracy", color="blue", linewidth=2)
    ax1.set_xlabel("Threshold", fontsize=14)
    ax1.set_ylabel("Balanced Accuracy", color="blue", fontsize=14)
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid()

    # Secondary y-axis for Disparate Impact and Average Odds Difference
    ax2 = ax1.twinx()
    ax2.plot(thresholds, test_disp_impacts, label="Disparate Impact", color="orange", linestyle="--", linewidth=2)
    ax2.plot(thresholds, test_avg_odds_diffs, label="Average Odds Difference", color="red", linestyle="-.", linewidth=2)

    ax2.set_ylabel("Fairness Metrics", color="red", fontsize=14)
    ax2.tick_params(axis='y', labelcolor="red")

    # Highlight the best threshold
    ax1.axvline(best_threshold, color='green', linestyle='--', linewidth=2, label="Best Threshold")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)

    # Title and layout adjustments
    fig.suptitle("Test Metrics vs Threshold (transformed test data)", fontsize=16)
    fig.tight_layout()
    plt.show()


def train_classifier_on_transformed_data(train_transf, test_transf, classifier_type, best_threshold, unprivileged_groups, privileged_groups):
    # Select classifier
    if classifier_type == 'logistic_regression':
        #classifier = LogisticRegression(random_state=1, solver='liblinear', max_iter=1000)   # depends on which logistic regression you use here
        classifier = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=1)
        #classifier = LogisticRegression(random_state=1)

    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=1)
    elif classifier_type == 'xgboost':
        classifier = XGBClassifier(random_state=1)
    else:
        raise ValueError("Invalid classifier type specified.")
    
    # Train the classifier on the transformed training data
    classifier.fit(train_transf.features, train_transf.labels.ravel(), sample_weight=train_transf.instance_weights)
    
    # Apply the best threshold directly to the test set (transformed)
    test_scores = classifier.predict_proba(test_transf.features)[:, 1]
    test_LR_predictions = (test_scores >= best_threshold).astype(int)

    # Create a copy of the test dataset and set predicted labels
    test_with_LR_scores = copy.deepcopy(test_transf)
    test_with_LR_scores.labels = test_LR_predictions.reshape(-1, 1)

    # Calculate fairness and performance metrics on the transformed test set
    test_metric = ClassificationMetric(test_transf, test_with_LR_scores,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)
    
    metrics = {
        'balanced_accuracy': (test_metric.true_positive_rate() + test_metric.true_negative_rate()) / 2,
        'statistical_parity_difference': test_metric.statistical_parity_difference(),
        'disparate_impact': test_metric.disparate_impact(),
        'average_odds_difference': test_metric.average_odds_difference(),
        'equal_opportunity_difference': test_metric.equal_opportunity_difference(),
        'theil_index': test_metric.theil_index()
    }

    # Print the metrics
    print(f"Balanced Accuracy (test transformed): {metrics['balanced_accuracy']}")
    print(f"Statistical Parity Difference (test transformed): {metrics['statistical_parity_difference']}")
    print(f"Disparate Impact (test transformed): {metrics['disparate_impact']}")
    print(f"Average Odds Difference (test transformed): {metrics['average_odds_difference']}")
    print(f"Equal Opportunity Difference (test transformed): {metrics['equal_opportunity_difference']}")
    print(f"Theil Index (test transformed): {metrics['theil_index']}")

    return classifier, test_scores, metrics

def standardize_features(train, val, test):
    # Standardizing the features
    scaler = StandardScaler()
    train.features = scaler.fit_transform(train.features)
    val.features = scaler.transform(val.features)
    test.features = scaler.transform(test.features)
    
    return train, val, test

def extract_fairness_metrics(
        
    train_before_stat_parity_diff, train_after_stat_parity_diff,
    test_before_stat_parity_diff, test_after_stat_parity_diff,
    train_before_disp_impact, train_after_disp_impact,
    test_before_disp_impact, test_after_disp_impact,
    best_threshold, best_balanced_acc,
    test_metrics, test_transf_metrics
):
    # Extract the fairness metrics before and after transformation from the given test metrics
    balanced_accuracy = test_metrics['balanced_accuracy']
    statistical_parity_difference = test_metrics['statistical_parity_difference']
    disparate_impact = test_metrics['disparate_impact']
    average_odds_difference = test_metrics['average_odds_difference']
    equal_opportunity_difference = test_metrics['equal_opportunity_difference']
    theil_index = test_metrics['theil_index']

    # Extract the fairness metrics for transformed data
    balanced_accuracy_transf = test_transf_metrics['balanced_accuracy']
    statistical_parity_difference_transf = test_transf_metrics['statistical_parity_difference']
    disparate_impact_transf = test_transf_metrics['disparate_impact']
    average_odds_difference_transf = test_transf_metrics['average_odds_difference']
    equal_opportunity_difference_transf = test_transf_metrics['equal_opportunity_difference']
    theil_index_transf = test_transf_metrics['theil_index']

    # Organizing the metrics with actual calculated values
    results_data = {
        "Dataset": [
            "Train", "Train", "Test", "Test", 
            "Train", "Train", "Test", "Test", 
            "Validation", "Validation", 
            "Test (Original)", "Test (Transformed)",
            "Test (Original)", "Test (Transformed)",
            "Test (Original)", "Test (Transformed)",
            "Test (Original)", "Test (Transformed)",
            "Test (Original)", "Test (Transformed)",
            "Test (Original)", "Test (Transformed)"
        ],
        "Stage": [
            "Before Transformation", "After Transformation", 
            "Before Transformation", "After Transformation",
            "Before Transformation", "After Transformation", 
            "Before Transformation", "After Transformation",
            "Best Threshold (Validation)", "Best Threshold (Validation)",
            "After Threshold", "After Threshold", 
            "After Threshold", "After Threshold", 
            "After Threshold", "After Threshold", 
            "After Threshold", "After Threshold", 
            "After Threshold", "After Threshold", 
            "After Threshold", "After Threshold"
        ],
        "Metric": [
            "Statistical Parity Difference", "Statistical Parity Difference",
            "Statistical Parity Difference", "Statistical Parity Difference",
            "Disparate Impact", "Disparate Impact",
            "Disparate Impact", "Disparate Impact",
            "Threshold", "Balanced Accuracy",
            "Balanced Accuracy", "Balanced Accuracy", 
            "Statistical Parity Difference", "Statistical Parity Difference", 
            "Disparate Impact", "Disparate Impact",
            "Average Odds Difference", "Average Odds Difference",
            "Equal Opportunity Difference", "Equal Opportunity Difference", 
            "Theil Index", "Theil Index"
        ],
        "Value": [
            float(train_before_stat_parity_diff),  # Statistical Parity Difference for Train (Before)
            float(train_after_stat_parity_diff),   # Statistical Parity Difference for Train (After)
            float(test_before_stat_parity_diff),  # Statistical Parity Difference for Test (Before)
            float(test_after_stat_parity_diff),   # Statistical Parity Difference for Test (After)
            float(train_before_disp_impact),      # Disparate Impact for Train (Before)
            float(train_after_disp_impact),       # Disparate Impact for Train (After)
            float(test_before_disp_impact),       # Disparate Impact for Test (Before)
            float(test_after_disp_impact),         # Disparate Impact for Test (After)
            best_threshold,
            best_balanced_acc,
            balanced_accuracy, 
            balanced_accuracy_transf,
            statistical_parity_difference, 
            statistical_parity_difference_transf,
            disparate_impact, 
            disparate_impact_transf,
            average_odds_difference, 
            average_odds_difference_transf,
            equal_opportunity_difference, 
            equal_opportunity_difference_transf,
            theil_index, 
            theil_index_transf
        ]
    }

    # Create the DataFrame
    results_table = pd.DataFrame(results_data)

    # Separate the data into three groups
    fairness_metrics = results_table.iloc[0:8].reset_index(drop=True)
    validation_metrics = results_table.iloc[8:10].reset_index(drop=True)
    test_metrics = results_table.iloc[10:22].reset_index(drop=True)

    # Name the tables
    fairness_metrics.name = "Fairness Metrics Before and After Transformation"
    validation_metrics.name = "Validation Metrics for Threshold Selection"
    test_metrics.name = "Test Metrics After Applying Threshold On Original and Transformed"

    # Display the tables with their names
    return fairness_metrics, validation_metrics, test_metrics


def train_classifier_with_dir(train, test, classifier_type, best_threshold, unprivileged_groups, privileged_groups, sensitive_attribute="sex", repair_levels=np.linspace(0., 1., 11)):
    
    # Select classifier
    if classifier_type == 'logistic_regression':
        #classifier = LogisticRegression(random_state=1, solver='liblinear', max_iter=1000)   # depends on which logistic regression you use here
        classifier = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=1)
        #classifier = LogisticRegression(random_state=1)

    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=1)
    elif classifier_type == 'xgboost':
        classifier = XGBClassifier(random_state=1)
    else:
        raise ValueError("Invalid classifier type specified.")
    
    
    indexOfSensitiveAttribute = train.feature_names.index(sensitive_attribute)
    results = []

    for repair_level in repair_levels:
        print(f"\n=== Repair Level: {repair_level} ===")
        
        # Apply Disparate Impact Remover with the current repair level
        disparate_impact_remover = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute="sex")
        train_transf = disparate_impact_remover.fit_transform(train)
        test_transf = disparate_impact_remover.fit_transform(test)
        
        X_train = train_transf.features
        X_train_without_sensitive_attribute = np.delete(X_train, indexOfSensitiveAttribute, axis=1)
        y_train = train_transf.labels.ravel()

        # Train a classifier on transformed training data
        classifier.fit(X_train_without_sensitive_attribute, y_train)

        X_test_transf= test_transf.features
        X_test_transf_without_sensitive_attribute = np.delete(X_test_transf,
                                                indexOfSensitiveAttribute,
                                                axis=1)
        # Apply the best threshold to the transf test set
        test_transf_scores = classifier.predict_proba(X_test_transf_without_sensitive_attribute)[:, 1] #only for the favorable class?
        test_transf_LR_predictions= (test_transf_scores >= best_threshold).astype(int)

        # Create a copy of the test dataset and set predicted labels
        test_transf_with_LR_scores = copy.deepcopy(test_transf)
        test_transf_with_LR_scores.labels = test_transf_LR_predictions.reshape(-1, 1)

        # Calculate fairness and performance metrics on the test set
        test_transf_metric = ClassificationMetric(test_transf, test_transf_with_LR_scores,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)

            # Store results for current repair level

        true_positive_rate = test_transf_metric.true_positive_rate()
        true_negative_rate = test_transf_metric.true_negative_rate()
        balanced_accuracy_transf = (true_positive_rate + true_negative_rate) / 2
        statistical_parity_difference_transf = test_transf_metric.statistical_parity_difference()
        disparate_impact_transf = test_transf_metric.disparate_impact()
        average_odds_difference_transf = test_transf_metric.average_odds_difference()
        equal_opportunity_difference_transf = test_transf_metric.equal_opportunity_difference()
        theil_index_transf = test_transf_metric.theil_index()



        results.append({
            "Repair Level": repair_level,
            "Balanced Accuracy": balanced_accuracy_transf,
            "Statistical Parity Difference": statistical_parity_difference_transf,
            "Disparate Impact": disparate_impact_transf,
            "Average Odds Difference": average_odds_difference_transf,
            "Equal Opportunity Difference": equal_opportunity_difference_transf,
            "Theil Index": theil_index_transf
        })
    
    # Return results as a DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    print("\nResults for Various Repair Levels:")
    print(results_df)

    return results_df

def apply_DIR(train, test, chosen_repair_level, classifier_type, best_threshold, unprivileged_groups, privileged_groups, sensitive_attribute="sex"):


    repair_level = chosen_repair_level


    # Select classifier
    if classifier_type == 'logistic_regression':
        #classifier = LogisticRegression(random_state=1, solver='liblinear', max_iter=1000)   # depends on which logistic regression you use here
        classifier = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=1)
        #classifier = LogisticRegression(random_state=1)

    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=1)
    elif classifier_type == 'xgboost':
        classifier = XGBClassifier(random_state=1)
    else:
        raise ValueError("Invalid classifier type specified.")
    
    
    indexOfSensitiveAttribute = train.feature_names.index(sensitive_attribute)
    results = []


    print(f"\n=== Repair Level: {repair_level} ===")
    
    # Apply Disparate Impact Remover with the current repair level
    disparate_impact_remover = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute="sex")
    train_transf = disparate_impact_remover.fit_transform(train)
    test_transf = disparate_impact_remover.fit_transform(test)
    
    X_train = train_transf.features
    X_train_without_sensitive_attribute = np.delete(X_train, indexOfSensitiveAttribute, axis=1)
    y_train = train_transf.labels.ravel()

    # Train a classifier on transformed training data
    classifier.fit(X_train_without_sensitive_attribute, y_train)

    X_test_transf= test_transf.features
    X_test_transf_without_sensitive_attribute = np.delete(X_test_transf,
                                            indexOfSensitiveAttribute,
                                            axis=1)
    # Apply the best threshold to the transf test set
    test_transf_scores = classifier.predict_proba(X_test_transf_without_sensitive_attribute)[:, 1] #only for the favorable class?
    test_transf_LR_predictions= (test_transf_scores >= best_threshold).astype(int)

    # Create a copy of the test dataset and set predicted labels
    test_transf_with_LR_scores = copy.deepcopy(test_transf)
    test_transf_with_LR_scores.labels = test_transf_LR_predictions.reshape(-1, 1)

    # Calculate fairness and performance metrics on the test set
    test_transf_metric = ClassificationMetric(test_transf, test_transf_with_LR_scores,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

        # Store results for current repair level

    true_positive_rate = test_transf_metric.true_positive_rate()
    true_negative_rate = test_transf_metric.true_negative_rate()
    balanced_accuracy_transf = (true_positive_rate + true_negative_rate) / 2
    statistical_parity_difference_transf = test_transf_metric.statistical_parity_difference()
    disparate_impact_transf = test_transf_metric.disparate_impact()
    average_odds_difference_transf = test_transf_metric.average_odds_difference()
    equal_opportunity_difference_transf = test_transf_metric.equal_opportunity_difference()
    theil_index_transf = test_transf_metric.theil_index()



    results.append({
        "Repair Level": repair_level,
        "Balanced Accuracy": balanced_accuracy_transf,
        "Statistical Parity Difference": statistical_parity_difference_transf,
        "Disparate Impact": disparate_impact_transf,
        "Average Odds Difference": average_odds_difference_transf,
        "Equal Opportunity Difference": equal_opportunity_difference_transf,
        "Theil Index": theil_index_transf
    })

    metrics = {
    'balanced_accuracy': balanced_accuracy_transf,
    'statistical_parity_difference': statistical_parity_difference_transf,
    'disparate_impact': disparate_impact_transf,
    'average_odds_difference': average_odds_difference_transf,
    'equal_opportunity_difference': equal_opportunity_difference_transf,
    'theil_index': theil_index_transf
}

    # Return results as a DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    print("\nResults for Various Repair Levels:")
    print(results_df)

    return classifier, test_transf_scores, metrics

def plot_metrics_dir(test, test_scores, best_threshold, unprivileged_groups, privileged_groups, thresholds=np.arange(0.01, 1, 0.01)):

    test_balanced_accs = []
    test_disp_impacts = []
    test_avg_odds_diffs = []

    for threshold in thresholds:
        test_predictions = (test_scores >= threshold).astype(int)
        test_with_scores = copy.deepcopy(test)
        test_with_scores.labels = test_predictions.reshape(-1, 1)

        # Compute fairness and performance metrics
        test_metric = ClassificationMetric(test, test_with_scores, unprivileged_groups, privileged_groups)
        
        test_balanced_accs.append((test_metric.true_positive_rate() + test_metric.true_negative_rate()) / 2)
        test_disp_impacts.append(test_metric.disparate_impact())
        test_avg_odds_diffs.append(test_metric.average_odds_difference())

    # Plot Balanced Accuracy and Fairness Metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Balanced Accuracy
    ax1.plot(thresholds, test_balanced_accs, label="Balanced Accuracy", color="blue", linewidth=2)
    ax1.set_xlabel("Threshold", fontsize=14)
    ax1.set_ylabel("Balanced Accuracy", color="blue", fontsize=14)
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid()

    # Secondary y-axis for Disparate Impact and Average Odds Difference
    ax2 = ax1.twinx()
    ax2.plot(thresholds, test_disp_impacts, label="Disparate Impact", color="orange", linestyle="--", linewidth=2)

        
    ax2.set_ylim(-0.05, 2)
    ax2.set_ylabel("Fairness Metrics", color="red", fontsize=14)
    ax2.tick_params(axis='y', labelcolor="red")

    # Highlight the best threshold
    ax1.axvline(best_threshold, color='green', linestyle='--', linewidth=2, label="Best Threshold")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)

    # Title and layout adjustments
    fig.suptitle("Test Metrics vs Threshold (transformed test data)", fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_metrics_aod(test, test_scores, best_threshold, unprivileged_groups, privileged_groups, thresholds=np.arange(0.01, 1, 0.01)):

    test_balanced_accs = []
    test_disp_impacts = []
    test_avg_odds_diffs = []

    for threshold in thresholds:
        test_predictions = (test_scores >= threshold).astype(int)
        test_with_scores = copy.deepcopy(test)
        test_with_scores.labels = test_predictions.reshape(-1, 1)

        # Compute fairness and performance metrics
        test_metric = ClassificationMetric(test, test_with_scores, unprivileged_groups, privileged_groups)
        
        test_balanced_accs.append((test_metric.true_positive_rate() + test_metric.true_negative_rate()) / 2)
        test_disp_impacts.append(test_metric.disparate_impact())
        test_avg_odds_diffs.append(test_metric.average_odds_difference())

    # Plot Balanced Accuracy and Fairness Metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Balanced Accuracy
    ax1.plot(thresholds, test_balanced_accs, label="Balanced Accuracy", color="blue", linewidth=2)
    ax1.set_xlabel("Threshold", fontsize=14)
    ax1.set_ylabel("Balanced Accuracy", color="blue", fontsize=14)
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid()

    # Secondary y-axis for Disparate Impact and Average Odds Difference
    ax2 = ax1.twinx()
    ax2.plot(thresholds, test_avg_odds_diffs, label="Average Odds Difference", color="red", linestyle="-.", linewidth=2)

    ax2.set_ylim(-0.5, 0.5)
    ax2.set_ylabel("Fairness Metrics", color="red", fontsize=14)
    ax2.tick_params(axis='y', labelcolor="red")

    # Highlight the best threshold
    ax1.axvline(best_threshold, color='green', linestyle='--', linewidth=2, label="Best Threshold")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)

    # Title and layout adjustments
    fig.suptitle("Test Metrics vs Threshold (transformed test data)", fontsize=16)
    fig.tight_layout()
    plt.show()


#Adult synthetic
LICENSE_KEY = "licensekeyoverride2023@"

def aif360_to_clearbox(aif360_dataset, target_column):

    df, _ = aif360_dataset.convert_to_dataframe()

    if 'Income Binary' in df.columns:
        df = df.rename(columns={'Income Binary': 'income'})

    return Dataset(
        data=df,
        target_column=target_column,
        regression=False,  
        name="Converted AIF360 Dataset"
    )


def filter_dataset(dataset, filter_type):
    features_array = dataset.get_x()
    labels_array = dataset.get_y()
    features_df = pd.DataFrame(features_array, columns=dataset.x_columns())
    # Determine the filter mask based on the filter_type
    if filter_type == "positive_women":
        mask = (features_df["sex"] == 0) & (labels_array == 1)
    elif filter_type == "positive_women_unprivileged_race":
        mask = (features_df["sex"] == 0) & (features_df["race"] == 0) & (labels_array == 1)
    elif filter_type == "positive_women_above_70":
        mask = (features_df["sex"] == 0) & (features_df["Age (decade)=>=70"] == 1) & (labels_array == 1)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    filtered_features = features_df.loc[mask]
    filtered_labels = pd.Series(labels_array).loc[mask].reset_index(drop=True)
    combined_df = pd.concat([filtered_features.reset_index(drop=True), filtered_labels.rename("income")], axis=1)
    return Dataset(
        data=combined_df,
        column_types=dataset.column_types,
        target_column="income",
        regression=dataset.regression
    ), combined_df

def generate_multiple_synthetic_datasets(
    dataset_df, 
    target_column, 
    engine_class, 
    num_datasets, 
    epochs=5, 
    learning_rate=0.001, 
    half=False, 
    extra_percentage=0.5
):

    dataset = Dataset(
        data=dataset_df,
        target_column=target_column,
        column_types=None,
        regression=False,
    )
    preprocessor = Preprocessor(dataset)
    X = preprocessor.transform(dataset.get_x())

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(dataset.get_y()).reshape(-1, 1)  # Convert to numerical and reshape
    
    # Ensure all data types are numerical
    X = X.astype(float)
    Y = Y.astype(float)

    engine = engine_class(
        license_key=LICENSE_KEY,
        layers_size=[50],
        x_shape=X.shape[1:],
        y_shape=Y.shape[1:],
        ordinal_feature_sizes=preprocessor.get_features_sizes()[0],
        categorical_feature_sizes=preprocessor.get_features_sizes()[1],
    )

    print("Training Tabular Engine...")
    engine.fit(X, y_train_ds=Y, epochs=epochs, learning_rate=learning_rate)

    synthesizer = LabeledSynthesizer(dataset, engine)

    # Generate the main synthetic datasets
    synthetic_data_list = [synthesizer.generate(has_header=True) for _ in range(num_datasets)]
    
    # Add a partial synthetic dataset if `half` is True
    if half:
        extra_data_size = int(len(dataset_df) * extra_percentage)
        print(f"Generating an extra partial synthetic dataset of size: {extra_data_size}")
        extra_synthetic_data = synthesizer.generate(has_header=True).sample(n=extra_data_size, random_state=42)
        synthetic_data_list.append(extra_synthetic_data)

    # Combine all synthetic datasets
    concatenated_synthetic_data = pd.concat(synthetic_data_list, axis=0, ignore_index=True)
    
    return concatenated_synthetic_data
    
