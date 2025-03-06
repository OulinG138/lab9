# Learning goals

In this lab, you are expected to practice the following skills:

- Evaluate whether a problem can be parallelized or not.
- Practice with the parallel package.
- Use Rscript to submit jobs.

## Problem 1

Give yourself a few minutes to think about what you learned about
parallelization. List three examples of problems that you believe may be
solved using parallel computing, and check for packages on the HPC CRAN
task view that may be related to it.

_Answer here._ Cross-validation in machine learning, caret -&gt;
supports parallel cross-validation with `doParallel` mlr, foreach,
doParallel -&gt; for parallel model training.

bootstrapping boot -&gt; for bootstrapping parallel -&gt; parallelize
resampling

markov chain monte carlo parallel rstan -&gt; for stan for bayesian
modeling RcppParallel -&gt;parallel mcmc sampling nimle -&gt; customize
bayesian inference

## Problem 2: Pre-parallelization

The following functions can be written to be more efficient without
using `parallel`:

1.  This function generates a `n x k` dataset with all its entries
    having a Poisson distribution with mean `lambda`.

<!-- -->

    fun1 <- function(n = 100, k = 4, lambda = 4) {
      x <- NULL

      for (i in 1:n)
        x <- rbind(x, rpois(k, lambda))

      return(x)
    }

    fun1alt <- function(n = 100, k = 4, lambda = 4) {
      # YOUR CODE HERE
      matrix(rpois(n*k, lambda = lambda),ncol = k)
    }

    # Benchmarking
    microbenchmark::microbenchmark(
      fun1(100),
      fun1alt(100),
      unit = "ns"
    )

    ## Warning in microbenchmark::microbenchmark(fun1(100), fun1alt(100), unit =
    ## "ns"): less accurate nanosecond times to avoid potential integer overflows

    ## Unit: nanoseconds
    ##          expr    min       lq      mean   median       uq     max neval
    ##     fun1(100) 104345 109326.5 174599.73 111417.5 115968.5 6185506   100
    ##  fun1alt(100)   9061   9758.0  16814.51  10086.0  10721.5  643946   100

How much faster?

_Answer here._ The new function is a lot faster for all mean, median,
min, max.

1.  Find the column max (hint: Checkout the function `max.col()`).

<!-- -->

    # Data Generating Process (10 x 10,000 matrix)
    set.seed(1234)
    x <- matrix(rnorm(1e4), nrow=10)

    # Find each column's max value
    fun2 <- function(x) {
      apply(x, 2, max)
    }

    fun2alt <- function(x) {
      # YOUR CODE HERE
      x[cbind(max.col(t(x)), 1:ncol(x))]
    }

    # Benchmarking
    bench <- microbenchmark::microbenchmark(
      fun2(x),
      fun2alt(x),
      unit = "ns"
    )
    plot(bench)
    ggplot2::autoplot(bench) + ggplot2::theme_minimal()

![](Lab09_files/figure-markdown_strict/p2-fun2-1.png)![](Lab09_files/figure-markdown_strict/p2-fun2-2.png)

_Answer here with a plot._

## Problem 3: Parallelize everything

We will now turn our attention to non-parametric
[bootstrapping](<https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>).
Among its many uses, non-parametric bootstrapping allow us to obtain
confidence intervals for parameter estimates without relying on
parametric assumptions.

The main assumption is that we can approximate many experiments by
resampling observations from our original dataset, which reflects the
population.

This function implements the non-parametric bootstrap:

    my_boot <- function(dat, stat, R, ncpus = 1L) {

      # Getting the random indices
      n <- nrow(dat)
      idx <- matrix(sample.int(n, n*R, TRUE), nrow=n, ncol=R)

      # Making the cluster using `ncpus`
      # STEP 1: GOES HERE
      cl <- makePSOCKcluster(ncpus)
      # STEP 2: GOES HERE
      # on.exit(stopCluster(cl))
      clusterExport(cl, varlist = c("idx","dat", "stat"), envir=environment())


      # STEP 3: THIS FUNCTION NEEDS TO BE REPLACED WITH parLapply
      ans <- parLapply(cl,seq_len(R), function(i) {
        stat(dat[idx[,i], , drop=FALSE])
      })

      # Coercing the list into a matrix
      ans <- do.call(rbind, ans)

      # STEP 4: GOES HERE
      stopCluster(cl)
      ans
    }

1.  Use the previous pseudocode, and make it work with `parallel`. Here
    is just an example for you to try:

<!-- -->

    library(parallel)
    # Bootstrap of a linear regression model
    my_stat <- function(d) coef(lm(y~x, data = d))

    # DATA SIM
    set.seed(1)
    n <- 500
    R <- 1e4
    x <- cbind(rnorm(n))
    y <- x*5 + rnorm(n)

    # Check if we get something similar as lm
    ans0 <- confint(lm (y~x))
    cat("OLS CI \n ")

    ## OLS CI
    ##

    print(ans0)

    ##                  2.5 %     97.5 %
    ## (Intercept) -0.1379033 0.04797344
    ## x            4.8650100 5.04883353

    ans1 <- my_boot(dat = data.frame(x,y), my_stat, R = R, ncpus = 4)
    qs <- c(.025, .975)
    cat("Bootstrap CI \n")

    ## Bootstrap CI

    print(t(apply(ans1, 2, quantile, probs = qs)))

    ##                   2.5%      97.5%
    ## (Intercept) -0.1386903 0.04856752
    ## x            4.8685162 5.04351239

1.  Check whether your version actually goes faster than the
    non-parallel version:

<!-- -->

    # your code here
    parallel::detectCores()

    ## [1] 11

    system.time(my_boot(dat = data.frame(x,y), my_stat, R = 4000, ncpus = 1L))

    ##    user  system elapsed
    ##   0.025   0.005   1.051

    system.time(my_boot(dat = data.frame(x,y), my_stat, R = 4000, ncpus = 3L))

    ##    user  system elapsed
    ##   0.041   0.011   0.506

_Answer here._  
user system elapsed 0.024 0.005 1.015 user system elapsed 0.041 0.011
0.499 The elapsed time is shorter with the parallel version.

## Problem 4: Compile this markdown document using Rscript

Once you have saved this Rmd file, try running the following command in
your terminal:

    Rscript --vanilla -e 'rmarkdown::render("/Users/zhangkaiwen/370/lab9/lab09-hpc.Rmd")' &

Where `[full-path-to-your-Rmd-file.Rmd]` should be replace with the full
path to your Rmd file… :).
