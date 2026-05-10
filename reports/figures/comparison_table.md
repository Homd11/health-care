# Clustering algorithm comparison

|                |   silhouette |   davies_bouldin |   calinski_harabasz |   mean_ari |   std_ari |   n_clusters |
|:---------------|-------------:|-----------------:|--------------------:|-----------:|----------:|-------------:|
| KMeans         |        0.953 |            0.027 |             328.29  |      0.8   |     0.4   |            2 |
| Agglo-Ward     |        0.953 |            0.027 |             328.29  |      0.8   |     0.4   |            2 |
| Agglo-Complete |        0.953 |            0.027 |             328.29  |    nan     |   nan     |            2 |
| Agglo-Average  |        0.953 |            0.027 |             328.29  |    nan     |   nan     |            2 |
| DBSCAN         |      nan     |          nan     |             nan     |      0.988 |     0.037 |            1 |
| GMM            |        0.807 |            0.621 |             545.354 |      0.904 |     0.119 |            5 |