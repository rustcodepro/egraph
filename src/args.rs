use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "egraph",
    version = "1.0",
    about = "autoencoder for variant annotation.
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// bactgraph
    Classifier {
        /// inputfile
        filepathinput: String,
        /// quality
        qualityinput: String,
        /// genotype prediction
        genotypeinput: String,
    },
    /// filter and classify the variants on the specific variant types
    VariantClassifier {
        /// file to be used
        fileinput: String,
        /// variant to be filtered
        variantinput: String,
        /// quality value
        qualityvalueinput: String,
        /// genotype prediction
        genotypepathinput: String,
    },
}
