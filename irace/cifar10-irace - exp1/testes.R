install.packages('tidyverse')
install.packages('irace')
library('irace')
library('tidyverse')
load("irace.Rdata")
head(iraceResults$allConfigurations)
print(iraceResults$allElites)
print(iraceResults$iterationElites)


last <- length(iraceResults$iterationElites)
id <- iraceResults$iterationElites[last]
getConfigurationById(iraceResults, ids = id)

# As an example, we use the best configuration found
best.config <- getFinalElites(iraceResults = iraceResults)
id <- best.config$.ID.
# Obtain the configurations using the identifier
# of the best configuration
all.exp <- iraceResults$experiments[,as.character(id)]
all.exp[!is.na(all.exp)]

# As an example, we get seed and instance of the experiments
# of the best candidate.
# Get index of the instances
pair.id <- which(!is.na(all.exp))
index <- iraceResults$state$.irace$instancesList[pair.id, "instance"]

#Ver um parâmetro específico das elites
iraceResults$state$model["camadasConvolucionais"] 

results <- iraceResults$experiments
# Wilcoxon paired test
conf <- gl(ncol(results), # number of configurations
           nrow(results), # number of instances
           labels = colnames(results))
pairwise.wilcox.test (as.vector(results), conf, paired = TRUE, p.adj = "bonf") 

irace:::concordance(results)
# Mesmo para a elite
# Wilcoxon paired test
conf <- gl(ncol(all.exp), # number of configurations
           nrow(all.exp), # number of instances
           labels = colnames(all.exp))
pairwise.wilcox.test (as.vector(all.exp), conf, paired = TRUE, p.adj = "bonf") 

irace:::concordance(all.exp)

#plot
# Get number of iterations
iters <- unique(iraceResults$experimentLog[, "iteration"])
# Get number of experiments (runs of target-runner) up to each iteration
fes <- cumsum(table(iraceResults$experimentLog[,"iteration"]))
# Get the mean value of all experiments executed up to each iteration
# for the best configuration of that iteration.
elites <- as.character(iraceResults$iterationElites)
values <- colMeans(iraceResults$experiments[, elites])
stderr <- function(x) sqrt(var(x)/length(x))
err <- apply(iraceResults$experiments[, elites], 2, stderr)
plot(fes, values, type = "s",
     xlab = "Number of runs of the target algorithm",
     ylab = "Mean value over testing set")
#ylab = "Mean value over testing set", ylim=c(23000000,23500000))
points(fes, values, pch=19)
arrows(fes, values - err, fes, values + err, length=0.05, angle=90, code=3)
text(fes, values, elites, pos = 1)

allConfs = iraceResults$allConfigurations
logs = iraceResults$experimentLog
summary(logs[time])

write.csv(results, "results.csv")
write.csv(best.config, "bestconfig.csv")
write.csv(allConfs, "allconfig.csv")
write.csv(logs, "logs.csv")
