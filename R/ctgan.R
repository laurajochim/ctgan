#' Initialize a CTGAN Model
#'
#' Initializes a CTGAN model object
#'
#' @import forge
#' @param embedding_dim Dimension of embedding layer.
#' @param generator_dim Dimensions of generator layers.
#' @param discriminator_dim Dimensions of discriminator layers.
#' @param generator_decay ADAM weight decay.
#' @param discriminator_decay description
#' @param batch_size Batch size.
#' @export
ctgan <- function(embedding_dim = 128, generator_dim = c(256, 256),
                  discriminator_dim = c(256, 256), generator_decay = 1e-6, discriminator_decay = 1e-6, batch_size = 500) {
  embedding_dim <- as.integer(embedding_dim)
  generator_dim <- as.integer(generator_dim)
  discriminator_dim <- as.integer(discriminator_dim)
  generator_decay <- as.double(generator_decay)
  discriminator_decay <- as.double(discriminator_decay)
  batch_size <- as.integer(batch_size)

  ctgan <- reticulate::import("ctgan")
  model <- ctgan$CTGAN(
    embedding_dim = embedding_dim,
    generator_dim = generator_dim,
    discriminator_dim = discriminator_dim,
    generator_decay = generator_decay,
    discriminator_decay = discriminator_decay,
    batch_size = batch_size
  )

  CTGANModel$new(model)
}

CTGANModel <- R6::R6Class(
  "CTGANModel",
  public = list(
    initialize = function(model_obj, metadata = NULL) {
      private$model_obj <- model_obj
      private$metadata <- metadata
    },
    fit = function(train_data, epochs, log_frequency) {
      c(train_data, metadata) %<-% transform_data(train_data)

      categorical_col_indices <- which(metadata$col_info$type == "nominal") - 1
      categorical_columns <- if (length(categorical_col_indices)) {
        reticulate::tuple(as.list(categorical_col_indices))
      } else {
        reticulate::tuple()
      }

      private$metadata <- metadata

      private$model_obj$fit(
        train_data = as.matrix(train_data),
        discrete_columns = categorical_columns,
        epochs = epochs,
        log_frequency = log_frequency
      )
    },
    sample = function(n) {
      if (is.null(private$metadata)) {
        stop("Metadata not found, consider fitting the model to data first.",
             call. = FALSE)
      }
      mat <- private$model_obj$sample(n = n)

      colnames(mat) <- private$metadata$col_info$variable

      mat %>%
        tibble::as_tibble() %>%
        purrr::imap_dfc(function(v, nm) {
          if (!is.null(lvls <- private$metadata$categorical_levels[[nm]])) {
            lvls[v + 1]
          } else {
            v
          }
        })
    },
    save = function(path) {
      path <- normalizePath(path)
      dir.create(path, recursive = TRUE, showWarnings = FALSE)
      saveRDS(private$metadata, file.path(path, "metadata.rds"))
      reticulate::py_save_object(private$model_obj, file.path(path, "model.pickle"))
      invisible(NULL)
    }
  ),
  private = list(
    model_obj = NULL,
    metadata = NULL
  )
)

#' Train a CTGAN Model
#'
#' @param object A `CTGANModel` object.
#' @param train_data Training data, should be a data frame.
#' @param epochs Number of epochs to train.
#' @param log_frequency Whether to use log frequency of categorical levels in
#'   conditional sampling. Defaults to `TRUE`.
#' @param ... Additional arguments, currently unused.
#'
#' @export
fit.CTGANModel <-
  function(object, train_data,
           epochs = 100,...) {
    epochs <- cast_scalar_integer(epochs)

    object$fit(train_data, epochs)

    invisible(NULL)
  }

#' Synthesize Data Using a CTGAN Model
#'
#' @param ctgan_model A fitted `CTGANModel` object.
#' @param n Number of rows to generate.
#'
#' @export
ctgan_sample <- function(ctgan_model, n = 100) {
  n <- cast_scalar_integer(n)
  ctgan_model$sample(n)
}

#' @export
print.CTGANModel <- function(x, ...) {
  cat("A CTGAN Model")
}
