rm(list = ls())
N<-30
xc1<-matrix(rnorm(2*N)*0.5, N, 2) + 2
xc2<-matrix(rnorm(2*N)*0.5, N, 2) + 4
plot(xc1[,1], xc1[,2], xlim=c(0,6), ylim=c(0,6), col='red')
par(new=TRUE)
plot(xc2[,1], xc2[,2], xlim=c(0,6), ylim=c(0,6), col='blue')

xall<-rbind(xc1, xc2)
yall<-rbind(matrix(0, ncol=1, nrow=N), matrix(1, ncol=1, nrow=N))

w<-c(0, 0, 0)
eta<-0.01
maxepocas<-100
tol<-1e-10