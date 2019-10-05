library(quantmod)
library(lubridate)
library(plyr)
library(glmnet)
library(PerformanceAnalytics)
library(ggplot2)
library(plotly)
library(parallel)

rm(list = ls())

t1 <- "1990-01-01"
v <- c("SPY","IEF")
P.list <- lapply(v, function(sym) get(getSymbols(sym,from = t1)) )

## ------------------------------------------------------------------------
sapply(P.list,dim)

## ------------------------------------------------------------------------
lapply(P.list, function(x)  first(date(x)) )

## ------------------------------------------------------------------------
P.list5 <- lapply(P.list, function(x) x[,5])
P.list6 <- lapply(P.list, function(x) x[,6])

## ------------------------------------------------------------------------
P5 <- na.omit(Reduce(function(...) merge(...),P.list5 ))
P6 <- na.omit(Reduce(function(...) merge(...),P.list6 ))

# adjust names
names(P5) <- names(P6) <- c("SPY","IEF")
names(P5) <- paste(names(P5),"vol",sep = "_")
summary(P5$VIX_vol)
P5$VIX_vol <- NULL

## ------------------------------------------------------------------------
R6 <- Return.calculate(P6)
# add rolling difference 
R6_roll <- R6 - rollapply(R6,25,mean)
names(R6_roll) <- paste(names(R6_roll),"_roll",sep="")

R <- na.omit(merge(R6,R6_roll,P5))
SPY_next <- stats::lag(R$SPY,-1)
names(SPY_next) <- "SPY_next"
R <- na.omit(merge(SPY_next,R))

## ------------------------------------------------------------------------
R$CHANGE_next <- 1
R$CHANGE_next[R$SPY_next < -0.01] <- -1
table(R$CHANGE)

# stack into a dataset rather than an xts object
ds <- data.frame(date = date(R),R)
rownames(ds) <- NULL
ds$SPY_next <- NULL # drop the next day return

# define features
features <- names(ds)[!names(ds) %in% c("date","CHANGE_next")]

## ------------------------------------------------------------------------
sum_change <- ddply(ds,"CHANGE_next",function(x)  apply(x[,features],2,function(y) mean(y)/sd(y)) )
(sum_change[[1]] - sum_change[[2]])/abs(sum_change[[2]])

library(xtable)
xtable(t(sum_change)[-1,],digits = 4)


##################################################################################################


###########################################
######## MACHINE LEARNING ANALYSIS ########
###########################################
weeks <- date(unique(floor_date(ds$date,"week")))
weeks <- c(weeks, last(weeks) + weeks(1))
ds_predict_multiple <- list()

for(seed in 1:100) {
  cat("THIS IS SEED", seed, "\n")
  W <- 50
  al <- 0.5 # net elastic
  ds_predict <- data.frame()
  ds_beta <- list()
  
  w_seq <- W:(length(weeks)-2)
  
  ds_predict_f <- function(w) {
    #cat("This is week ",w, " out of ",length(weeks),"\n")
    # training set consists of relatively 250 daily observations
    train.weeks <- weeks[(w-W+1):(w+1)]
    train.index <- which((ds$date > train.weeks[1]) & (ds$date <= train.weeks[W+1]))
    
    # the weekly is around 5 days
    test.weeks <-  weeks[w+1:2]
    test.index <- which((ds$date > test.weeks[1]) & (ds$date < test.weeks[2]))
    
    # drop the last obs from the train set to avoid leakage
    DS <- ds[train.index[-length(train.index)],]
    x_train <- model.matrix( ~ .-1, DS[,features])
    
    # use CV
    set.seed(seed)
    try_error <- try(lm <- cv.glmnet(x=x_train,y = as.factor(DS$CHANGE_next), intercept=FALSE,
                                     family =   "multinomial", alpha=al, nfolds=10,parallel = T),silent = TRUE)
    
    i <- 1
    while(inherits(try_error,"try-error")) {
      #cat("Error in CV","\n")
      try_error <- try(lm <- cv.glmnet(x=x_train,y = as.factor(DS$CHANGE_next),
                                       intercept=FALSE, family =   "multinomial", alpha=al, nfolds=10,
                                       parallel = T),silent = TRUE)
      i <- i + 1
      if (i == 10)
        lm <- lm
    }
    
    # assign the lambda
    best_lambda <- lm$lambda.min
    
    # find the optimal model
    lm.star = glmnet(x=x_train,y = as.factor(DS$CHANGE_next), intercept=FALSE ,
                     family =   "multinomial", alpha=al, lambda = best_lambda)
    
    # fit the test sample
    DS_test <- ds[test.index,]
    x_test <- model.matrix( ~ .-1, DS_test[,features])
    DS_predict <- predict(lm.star,x_test,type = "response")
    
    # stack in data
    DS_predict <- data.frame(DS_predict)
    names(DS_predict) <- c("dn","up")
    DS_predict$date <- ds[test.index,"date"]
    
    # finally keep track of the glmnet results in a list
    ds_beta <- c(ds_beta,list(lm.star))
    
    list(DS_predict = DS_predict,lm_list = list(lm.star))
  }
  
  mclapply_list <- mclapply(w_seq,ds_predict_f)
  ds_predict_l <- lapply(mclapply_list, function(x) x$DS_predict )
  ds_predict <- ldply(ds_predict_l,data.frame)
  ds_predict_multiple <- c(ds_predict_multiple,list(ds_predict))
}

ds_predict_ups <- llply(ds_predict_multiple,function(x) x[,"up"])
ds_predict_dns <- llply(ds_predict_multiple,function(x) x[,"dn"])


ds_predict_ups <- apply(Reduce(cbind,ds_predict_ups),1,mean)
ds_predict_dns <- apply(Reduce(cbind,ds_predict_dns),1,mean)
ds_predict <- data.frame(dn = ds_predict_dns,up = ds_predict_ups,date = ds_predict$date)
summary(ds_predict)


## ------------------------------------------------------------------------
ds2 <- merge(ds,ds_predict,by = "date")
updn <- data.frame(dn = ds2$dn,up =ds2$up)
rownames(updn) <- ds2$date
updn <- as.xts(updn)
updn_roll <- na.omit(rollapply(updn,25,mean))
names(updn_roll) <- c("dn_roll","up_roll")
ds.updn <- data.frame(date = date(updn_roll),updn_roll)
rownames(ds.updn) <- NULL
ds3 <- merge(ds2,ds.updn, by = "date")

## ---- warning=F----------------------------------------------------------
next_f <- function(x) c(x[-1],NA)
ds3$IEF_next <- next_f(ds3$IEF)
ds3$SPY_next <- next_f(ds3$SPY)
ds3$BENCHMARK <- with(ds3,0.6*SPY_next + 0.4*IEF_next)

plot_performance <- function(a) {
  
  ds3$PORT <- with(ds3, (up_roll >= a)*(SPY_next)  +  (up_roll < a)*IEF_next )
  
  # load ds3 into a ggplot friendly data
  ds_plot <- data.frame(Date = ds3$date, return = cumsum(ds3$PORT), Type = "Strategy")
  ds_plot <- rbind(ds_plot, 
                   data.frame(Date = ds3$date, return = cumsum(ds3$BENCHMARK), Type = "Benchmark"))
  ds_plot <- rbind(ds_plot, 
                   data.frame(Date = ds3$date, return = cumsum(ds3$SPY_next), Type = "SPY"))
  ds_plot <- rbind(ds_plot, 
                   data.frame(Date = ds3$date, return = cumsum(ds3$IEF_next), Type = "IEF"))
  ds_plot <- rbind(ds_plot, 
                   data.frame(Date = ds3$date, return = ds3$dn_roll, Type = "Probability Down"))
  
  ds_plot <- na.omit(ds_plot)
  
  p <- ggplot(ds_plot) + geom_line(aes(x = Date,y = return,colour = Type)) 
  p <- p + geom_abline(intercept =0,linetype = "dashed")
  p <- ggplotly(p,height =  500, width = 900)
  
  # also return the data for performance comparison
  ds_perf <- ds3[,c("IEF_next","SPY_next","BENCHMARK","PORT","SPY_next")]
  rownames(ds_perf) <- ds3$date
  ds_perf <- as.xts(ds_perf)
  
  list(plot_perf = p,data_perf = ds_perf)
}


#### ML EFFICIENT FRONTIER ####

# benchmark mean and risk
mean0 <- 252*mean(plot_performance(0.8)$data[,"BENCHMARK"],na.rm = T)
sd0 <- sqrt(252)*sd(plot_performance(0.8)$data[,"BENCHMARK"],na.rm = T)

a_seq <- rev(seq(0.80,1,length = 100))
performance_all <- lapply(a_seq,plot_performance)
performance_all <- lapply(performance_all,function(x) x$data_perf[,"PORT"])
mean_seq <- 252*sapply(performance_all,function(x) mean(x,na.rm = T))
sd_seq <- sqrt(252)*sapply(performance_all,function(x) sd(x,na.rm = T))

mean_seq <- mean_seq[sd_seq <= sd0 + 0.01]
sd_seq <- sd_seq[sd_seq <= sd0 + 0.01]


##################################################################################################


######################################
##### MVEF ANALYSIS ##################
######################################


MVEF_f <- function(kappa_risk) {
  weeks <- date(unique(floor_date(ds$date,"week")))
  weeks <- c(weeks, last(weeks) + weeks(1))
  
  W <- 50
  w_seq <- W:(length(weeks)-2)
  
  MV_f <- function(w,kappa_risk) {
    
    train.weeks <- weeks[(w-W+1):(w+1)]
    train.index <- which((ds$date > train.weeks[1]) & (ds$date <= train.weeks[W+1]))
    
    # the weekly is around 5 days
    test.weeks <-  weeks[w+1:2]
    test.index <- which((ds$date > test.weeks[1]) & (ds$date < test.weeks[2]))
    
    # drop the last obs from the train set to avoid leakage
    DS <- ds[train.index[-length(train.index)],]
    R_ds <- DS[,c("SPY","IEF")]
    
    
    d <- ncol(R_ds)
    S <- var(R_ds)
    M <- apply(R_ds,2,mean)
    V <- solve(S)
    e <- as.matrix(rep(1,d))
    alpha0 <- V%*%e/sum(V)
    B <- V%*%(diag(1,d) - e%*%t(alpha0))
    alpha1 <- B%*%M
    
    xi <- alpha0 + (1/kappa_risk)*alpha1
    
    
    DS_predict <- matrix(rep(xi,length(test.index)),length(test.index),2,byrow = T)
    DS_predict <- data.frame(DS_predict)
    names(DS_predict) <- paste(names(R_ds),"MV",sep = "_")
    DS_predict$date <-  ds[test.index,"date"]
    
    return(DS_predict)
  }
  
  lapply_list_mv <- lapply(w_seq,function(w) MV_f(w,kappa_risk))
  MV_W <- Reduce(rbind,lapply_list_mv)
  
  ## ------------------------------------------------------------------------
  ds2 <- merge(ds,MV_W,by = "date")
  ds3 <- ds2
  
  ## ---- warning=F----------------------------------------------------------
  next_f <- function(x) c(x[-1],NA)
  ds3$IEF_next <- next_f(ds3$IEF)
  ds3$SPY_next <- next_f(ds3$SPY)
  
  ds3$PORT <- with(ds3, SPY_MV*SPY_next  +  SPY_MV*IEF_next )
  port_mean <- 252*mean(ds3$PORT,na.rm = T)
  port_sd <- sqrt(252)*sd(ds3$PORT,na.rm = T)
  return(c(port_mean,port_sd))
  
}

MVEF_full <- function(kappa_risk){
  R_ds <- ds3[,c("SPY","IEF")]
  d <- ncol(R_ds)
  S <- var(R_ds)
  M <- apply(R_ds,2,mean)
  V <- solve(S)
  e <- as.matrix(rep(1,d))
  alpha0 <- V%*%e/sum(V)
  B <- V%*%(diag(1,d) - e%*%t(alpha0))
  alpha1 <- B%*%M
  xi <- alpha0 + (1/kappa_risk)*alpha1
  mu_p <- 252*t(xi)%*%M
  sig_p <- sqrt(252)*sqrt(t(xi)%*%S%*%xi)
  return(c(mu_p,sig_p))
}

kappa_seq <- c(seq(4,30,length = 100))

performance_all_MV <- lapply(kappa_seq,MVEF_f)
performance_all_MV <- Reduce(rbind,performance_all_MV)
performance_all_MV <- performance_all_MV[performance_all_MV[,2] <= max(sd_seq) & performance_all_MV[,2] >= min(sd_seq),]

mean_seq2 <- performance_all_MV[,1]
sd_seq2 <- performance_all_MV[,2]

kappa_seq2 <- c(seq(0,10,length = 100))
performance_all_MV2 <- t(sapply(kappa_seq2,MVEF_full))
performance_all_MV2 <- performance_all_MV2[performance_all_MV2[,2] <= max(sd_seq) & performance_all_MV2[,2] >= min(sd_seq),]

mean_seq3 <- performance_all_MV2[,1]
sd_seq3 <- performance_all_MV2[,2]

plot(mean_seq~sd_seq,pch = 20,cex = 0.5, col = 1, xlab = expression(sigma[p]), ylab = expression(mu[p])  )
lines(sd_seq,predict(loess(mean_seq~sd_seq)), lwd = 2, col = 1)
points(mean_seq2~sd_seq2, pch = 15, col = 2, cex = 0.75  )
lines(sd_seq2,predict(loess(mean_seq2~sd_seq2)), lwd = 2, col = 2,lty = 3)
lines(sd_seq3,predict(loess(mean_seq3~sd_seq3)), lwd = 2, col = 3,lty = 4)
points(sd0,mean0,pch = 3,lty = 2,col = 3)


ds_plot1 <- data.frame(M = mean_seq,S = sd_seq,Type = "ML OUT",Size = 1)
ds_plot2 <- data.frame(M = mean_seq2,S = sd_seq2,Type = "MV OUT", Size = 1)
ds_plot3 <- data.frame(M = mean_seq3,S = sd_seq3,Type = "MV IN",Size = 1)
ds_plot4 <- data.frame(M = mean0,S = sd0,Type = "Naive", Size = 2)

ds_plot <- rbind(ds_plot4,ds_plot1,ds_plot2,ds_plot3)
ds_plot$Type <- as.factor(ds_plot$Type)

p1 <-ggplot(data=ds_plot, aes(x=S, y=M, colour=Type,shape = Type)) + geom_point(size = 3)
p1 <- p1 +  geom_smooth() + xlab("Volatility") + ylab("Mean")
p1  <- p1 + guides(color = guide_legend(override.aes = list(linetype = 0)))
p1


