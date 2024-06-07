library(ggplot2)
library(data.table)

d <- fread("history.csv")
d
dm <- melt(d, id.vars = c("platform","year","size"))
dm

colnames(dm) <- c("platform","year","size","software","runtime")

ggplot(dm, aes(x=year, y=runtime, color=software)) +
  geom_line() + geom_point() + scale_y_log10() +
  facet_wrap(size ~ platform) + 
  theme_minimal()

