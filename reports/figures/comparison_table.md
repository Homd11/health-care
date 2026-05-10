# Clustering algorithm comparison

|                |   silhouette |   davies_bouldin |   calinski_harabasz |   mean_ari |   std_ari |   n_clusters |
|:---------------|-------------:|-----------------:|--------------------:|-----------:|----------:|-------------:|
| KMeans         |        0.303 |            1.649 |             170.27  |      0.858 |     0.197 |            3 |
| Agglo-Ward     |        0.26  |            1.948 |             147.419 |      0.603 |     0.152 |            3 |
| Agglo-Complete |        0.498 |            0.789 |              20.417 |    nan     |   nan     |            3 |
| Agglo-Average  |        0.503 |            0.678 |              20.017 |    nan     |   nan     |            3 |
| DBSCAN         |      nan     |          nan     |             nan     |      0.868 |     0.088 |            1 |
| GMM            |        0.251 |            1.959 |             132.978 |      0.831 |     0.14  |            3 |