# R packages for the offline labs (02-06, 07B, 08) and for building the site.
# Run once:  source("install.R")

pkgs <- c(
  "ggplot2",      # plots
  "randomForest", # random forests
  "glmnet",       # ridge / lasso (genomic prediction)
  "rpart",        # decision trees
  "nnet",         # simple neural networks
  "vegan",        # community ecology: diversity, Bray-Curtis, PERMANOVA
  "uwot",         # UMAP
  "cluster",      # silhouette scores
  "IRkernel"      # R kernel for Jupyter (needed to build the site)
)
new <- setdiff(pkgs, rownames(installed.packages()))
if (length(new)) install.packages(new)

# Register the R kernel with Jupyter (only needed to rebuild the lab outputs)
# IRkernel::installspec()

# Day 2 labs 07A and 07C (run in Colab):
# install.packages(c("keras3", "reticulate", "httr", "jsonlite"))
