import os
import time
import sys
import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ensure project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from persistence import ModelPersistence
from src.data.preprocess.bagging_boosting import load_data, ensure_dir_exists
from src.visualization.visualize import visualize_results, VISUALIZATION_DIR

# your from-scratch implementations
from src.models.bagging_boosting.core.bagging import BaggingClassifier as ScratchBagging
from src.models.bagging_boosting.core.boosting import AdaBoostClassifier as ScratchAdaBoost

def main():
    ensure_dir_exists(VISUALIZATION_DIR)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    persistence = ModelPersistence()

    train_ds, test_ds = load_data()
    X_train = train_ds.imgs.reshape(len(train_ds), -1) / 255.0
    y_train = train_ds.labels.flatten()
    X_test  = test_ds.imgs.reshape(len(test_ds), -1) / 255.0
    y_test  = test_ds.labels.flatten()

    results = {}

    # 1) Base Decision Tree
    base = DecisionTreeClassifier(max_depth=5, random_state=42)
    base.fit(X_train, y_train)
    y_pred = base.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results['Base Tree'] = acc
    persistence.save_model(base, 'base_tree', f"base_tree_{timestamp}.joblib")
    visualize_results(
        y_pred, y_test,                           # preds, true
        'Base Tree',                              # <-- model_name
        save_path=os.path.join(
            VISUALIZATION_DIR, f"{timestamp}_base_tree_cm.png"
        )
    )

    # 2) scikit-learn Bagging
    sk_bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50,
        random_state=42
    )
    sk_bag.fit(X_train, y_train)
    y_pred = sk_bag.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results['SK Bagging'] = acc
    persistence.save_model(sk_bag, 'bagging_dt', f"bagging_dt_{timestamp}.joblib")
    visualize_results(
        y_pred, y_test,
        'SK Bagging',
        save_path=os.path.join(
            VISUALIZATION_DIR, f"{timestamp}_bagging_dt_cm.png"
        )
    )

    # 3) scikit-learn AdaBoost
    sk_boost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=0.5,
        random_state=42
    )
    sk_boost.fit(X_train, y_train)
    y_pred = sk_boost.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results['SK AdaBoost'] = acc
    persistence.save_model(sk_boost, 'adaboost_dt', f"adaboost_dt_{timestamp}.joblib")
    visualize_results(
        y_pred, y_test,
        'SK AdaBoost',
        save_path=os.path.join(
            VISUALIZATION_DIR, f"{timestamp}_adaboost_dt_cm.png"
        )
    )

    # 4) From-scratch Bagging
    scratch_bag = ScratchBagging(
        base_estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50,
        random_state=42
    )
    scratch_bag.fit(X_train, y_train)
    y_pred = scratch_bag.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results['Scratch Bagging'] = acc
    persistence.save_model(
        scratch_bag, 'bagging_scratch', f"bagging_scratch_{timestamp}.joblib"
    )
    visualize_results(
        y_pred, y_test,
        'Scratch Bagging',
        save_path=os.path.join(
            VISUALIZATION_DIR, f"{timestamp}_bagging_scratch_cm.png"
        )
    )

    # 5) From-scratch AdaBoost
    scratch_boost = ScratchAdaBoost(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=0.5,
        random_state=42
    )
    scratch_boost.fit(X_train, y_train)
    y_pred = scratch_boost.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results['Scratch AdaBoost'] = acc
    persistence.save_model(
        scratch_boost, 'boosting_scratch', f"boosting_scratch_{timestamp}.joblib"
    )
    visualize_results(
        y_pred, y_test,
        'Scratch AdaBoost',
        save_path=os.path.join(
            VISUALIZATION_DIR, f"{timestamp}_boosting_scratch_cm.png"
        )
    )

    # Summary
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    for name, score in results.items():
        print(f"{name}: Accuracy = {score:.4f}")

if __name__ == "__main__":
    main()