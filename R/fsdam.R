#library(reticulate); library(FSDAM); dat=hvtn505tier1[,-1]; opt_numCode=2; opt_seed=1; opt_model="n"; opt_gpu=0; opt_k=100; opt_nEpochs=500; opt_constr="newpenalization"; opt_tuneParam=10; opt_penfun="mean"; opt_ortho=1; opt_earlystop="no"
fsdam <- function(dat, opt_numCode=ncol(dat), opt_seed=1, opt_model="n", opt_gpu=0, opt_k=100, opt_nEpochs=10000, opt_constr=c("newpenalization", "constrained", "none"), opt_tuneParam=10, opt_penfun="mean", opt_ortho=1, opt_earlystop="no", verbose=FALSE) {    

    if (!reticulate::py_module_available("torch")) stop("Python torch module is not available. Try restarting R, use a different Python installation by reticulate::use_python(path) before calling fsdam") 
#    reticulate::use_python("/app/easybuild/software/Python/3.7.4-foss-2016b/bin/python")
#    reticulate::py_config() # to check 
     
    python_path <- system.file("python", package = "FSDAMVAE") # modified
    ae <- import_from_path("autoencoders", path = python_path)  
    
    if (!is.data.frame(dat)) dat=data.frame(dat)    
    
    out=ae$fsdam$main(dat, as.integer(opt_numCode), as.integer(opt_seed), opt_model, as.integer(opt_gpu), as.integer(opt_k), as.integer(opt_nEpochs), opt_constr, opt_tuneParam, opt_penfun, opt_ortho, opt_earlystop, verbose)    
    for (i in 1:length(out)) names(out[[i]])=c("reconstruct_loss", "pen_loss", "y_pred", "code", "decoder_w", "decoder_b", "history")
    
    ret = list(
          data=dat
        , mse=sapply(1:opt_numCode, function (i) out[[i]]$reconstruct_loss)
        , pen.loss=sapply(1:opt_numCode, function (i) out[[i]]$pen_loss)
        , code=sapply(1:opt_numCode, function (i) out[[i]]$code$cpu()$numpy())
        , prediction=lapply(1:opt_numCode, function (i) out[[i]]$y_pred$cpu()$numpy()) # cpu() is needed if gpu is used
        , history=sapply(1:opt_numCode, function (i) out[[i]]$history)
    )
    colnames(ret$history)=paste0("component ", 1:ncol(ret$history))
    
    class(ret)=c("fsdam","list")
    ret        
}
    

plot.fsdam=function (x, which=c("mse", "history", "decoder.func", "scatterplot"), k=NULL, dim.1=NULL, dim.2=NULL, col.predict=2, ...) {
    which=match.arg(which)
    if (is.null(k)) k=1
    
    if (which=="mse") {
        plot(0:length(x$mse), c(1, x$mse), xlab="Number of components", ylab="Proportion of variability unexplained", type="b", xaxt="n", ...)
        axis(1, at=0:length(x$mse), labels=0:length(x$mse))
    } else if (which=="history") {
        mymatplot(x$history, legend.x=3, ...)
    } else if (which=="decoder.func") {
        for (i in 1:ncol(x$data)) {
            plot(x$code[,i], x$prediction[[k]][,i], ...)
        }
    } else if (which=="scatterplot") {
        plot(x$data[,dim.1], x$data[,dim.2], ...)
        points(x$predict[[k]][,dim.1], x$predict[[k]][,dim.2], col=col.predict, ...)
    }

}
