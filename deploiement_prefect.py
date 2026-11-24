from prefect import serve

from pipeline_prefect import flow_all, flow_evaluate, flow_train

if __name__ == "__main__":
    dep_all = flow_all.to_deployment(
        name="kmeans-all-deployment",
        interval=3600,  # toutes les 60 minutes (en secondes)
    )
    dep_train = flow_train.to_deployment(
        name="kmeans-train-deployment",
    )
    dep_evaluate = flow_evaluate.to_deployment(
        name="kmeans-evaluate-deployment",
    )

    serve(dep_all, dep_train, dep_evaluate)
