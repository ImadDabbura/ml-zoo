library(ISLR)
library(MASS)
library(boot)
library(leaps)
library(caTools)
library(e1071)
library(glmnet)
library(ggplot2)
attach(Boston)

lm_fit <- glm(medv ~ ., data = Boston)
cv_error <- cv.glm(Boston, lm_fit)

lm_fit <- glm(mpg ~ horsepower, data = Auto)
cv_error <- cv.glm(Auto, lm_fit)

cv_error_10 <- rep(0, 10)
for(i in 1:10){
  lm_fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
  cv_error_10[i] <- cv.glm(Auto, lm_fit, K = 10)$delta[1]
}
cv_error_10
plot(1:10, cv_error_10, 'b', xlab = "K", ylab = "10-fold cv error")

boot_fn <- function(data, index){
  return(coef(lm(mpg ~ horsepower, data = Auto, subset = index)))
}
boot_fn(Auto, 1:392)
summary(lm(mpg ~ horsepower, data = Auto))


set.seed(1)
Hitters <- na.omit(Hitters)
train <- sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)
test <- !train
reg_fit <- regsubsets(Salary ~ ., data = Hitters[train, ], nvmax = 19)
test_mat <- model.matrix(Salary ~ ., data = Hitters[test, ])
val_error <- rep(0, 19)
for(i in 1:19){
  coeff_ <- coef(reg_fit, id = i)
  pred <- test_mat[, names(coeff_)] %*% coeff_
  val_error[i] <- mean((pred - Hitters$Salary[test])^2)
}

plot(1:19, val_error, "b")

corrplot::corrplot(cor(Hitters[, -c(14:15, 20)]), method = "number")

#======#
predict.regsubsets <- function(object, newdata, id, ...){
  model_formula <- as.formula(object$call[[2]])
  model_mat <- model.matrix(model_formula, newdata)
  coeff_ <- coef(object, id = id)
  model_mat[, names(coeff_)] %*% coeff_
}
#====#
k <- 10
set.seed(1)
n <- nrow(Hitters)
p <- dim(Hitters)[2] - 1
folds <- sample(rep(1:k, length = n))

cv_error <- matrix(0, nrow = k, ncol = p, dimnames = list(NULL, paste(1:p)))
for(i in 1:k){
  reg_fit <- regsubsets(Salary ~ ., data = Hitters[folds != i, ], nvmax = p)
  for(j in 1:p){
    pred <- predict(reg_fit, newdata = Hitters[folds == i, ], id = j)
    cv_error[i, j] <- mean((pred - Hitters$Salary[folds == i]) ^ 2)
  }
}

cv_error_mean <- apply(cv_error, 2, mean)
plot(1:p, cv_error_mean, type = "b")


#===#
lambda_grid <- 10 ^ seq(10, -2, length = 100)
x <- model.matrix(Salary ~ ., data = Hitters)[, -1]
y <- Hitters$Salary
ridge.mod <- glmnet(x, y, alpha = 0, lambda = lambda_grid, )
cv.out <- cv.glmnet(x, y, alpha = 0)
cv.out$lambda.min
plot(cv.out)
predict(cv.out, type = "coefficients", s = cv.out$lambda.min)
x = 
  
  
set.seed(1)
train <- sample(1:nrow(Hitters), nrow(Hitters) / 2)
test <- -train
x <- model.matrix(Salary ~ ., data = Hitters)[, -1]
y <- Hitters$Salary
ridge.mod <- glmnet(x[train, ], y[train], alpha = 0, lambda = lambda_grid)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)  
plot(cv.out)
best_lambda <- cv.out$lambda.min
best_lambda
pred <- predict(ridge.mod, s = best_lambda, newx = x[test, ])
mean((pred - y[test]) ^ 2)
out <- glmnet(x, y, alpha = 0)
predict(out, type = "coefficients", s = best_lambda)























