
from pymldb import Connection
mldb = Connection("http://localhost/")


mldb.put('/v1/procedures/import_bench_train_1m', {
    "type": "import.text",
    "params": { 
        "dataFileUrl": "https://s3.amazonaws.com/benchm-ml--main/train-1m.csv",
        "outputDataset":"bench_train_1m",
        "runOnCreation": True
    }
})

mldb.put('/v1/procedures/import_bench_test', {
    "type": "import.text",
    "params": { 
        "dataFileUrl": "https://s3.amazonaws.com/benchm-ml--main/test.csv",
        "outputDataset":"bench_test",
        "runOnCreation": True
    }
})


mldb.put('/v1/procedures/benchmark', {
    "type": "classifier.experiment",
    "params": {
        "experimentName": "benchm_ml",
        "inputData": """
            select
                {* EXCLUDING(dep_delayed_15min)} as features,
                dep_delayed_15min = 'Y' as label
            from bench_train_1m
            """,
        "testingDataOverride":  """
            select
                {* EXCLUDING(dep_delayed_15min)} as features,
                dep_delayed_15min = 'Y' as label
            from bench_test
            """,
        "configuration": {
            "type": "boosting",
            "validation_split": 0,
            "min_iter": 100,
            "max_iter": 100,
            "weak_learner": {
                "type": "decision_tree",
                "max_depth": 10,
                "random_feature_propn": 1
            }
        },
        "modelFileUrlPattern": "file:///mldb_data/models/benchml_$runid.cls",       
        "mode": "boolean"
    }
})


import time

start_time = time.time()

result = mldb.post('/v1/procedures/benchmark/runs')

run_time = time.time() - start_time
auc = result.json()["status"]["folds"][0]["resultsTest"]["auc"]

print "\n\nAUC = %0.10f, time = %0.4f\n\n" % (auc, run_time)


