library(ggplot2)
library(dplyr)

# cat celeste-256x8x4-3026755.out | grep secs | cut -d' ' -f2,3 | sed 's/://g'  > times.tsv
# Note: you need to get a couple extra lines out of this.

setwd("/home/rgiordan/Documents/git_repos/CelesteDev.jl/parallelism")
times <- read.delim("times.tsv", header=FALSE, sep=" ", colClasses=c("factor", "numeric"))
names(times) <- c("objid", "time")

hist(log(times$time), 100)
qqnorm(log(times$time))

min_time <- 3
times_sorted <- data.frame(time=sort(times$time), index=(nrow(times):1) / nrow(times))
coeffs <- coefficients(lm(log10(index) ~ log10(time), filter(times_sorted, time > 10)))
alpha <- coeffs[2] - 1
const <- coeffs[1]
ggplot(filter(times_sorted, time > min_time)) +
  geom_point(aes(x=log10(time), y=log10(index))) +
  geom_abline(aes(slope=alpha + 1, intercept=const)) +
  ylab("Log of 1 - cumulative distribution") +
  ggtitle(sprintf("Tail of 1 - optimiazation time CDF\nAlpha = %f", alpha))

