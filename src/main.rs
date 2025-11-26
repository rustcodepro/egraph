mod args;
mod egraph;
use crate::args::CommandParse;
use crate::args::Commands;
use crate::egraph::knnclassification;
use crate::egraph::logisticclassification;
use crate::egraph::randomforestclassification;
use crate::egraph::variantclasslabel_knn;
use crate::egraph::variantclasslabel_logistic;
use crate::egraph::variantclasslabel_random;
use clap::Parser;

/*
Gaurav Sablok
codeprog@icloud.com
- a smartcore implementation of the node clustering where you ahve the defined
- nodes and you have the weights defined. It will estimate teh features from the
- provided nodes file and then will use the weights to implement the cluster.
*/

fn main() {
    let argparse = CommandParse::parse();
    match &argparse.command {
        Commands::Classifier {
            filepathinput,
            qualityinput,
            genotypeinput,
        } => {
            let command_logistic =
                logisticclassification(filepathinput, qualityinput, genotypeinput).unwrap();
            let command_knn =
                knnclassification(filepathinput, qualityinput, genotypeinput).unwrap();
            let command_random =
                randomforestclassification(filepathinput, qualityinput, genotypeinput).unwrap();
            println!(
                "The commands have been finished:{}\n{}\n{}\n",
                command_logistic, command_knn, command_random
            );
        }
        Commands::VariantClassifier {
            variantinput,
            qualityvalueinput,
            fileinput,
            genotypepathinput,
        } => {
            let command_knn = variantclasslabel_knn(
                fileinput,
                variantinput,
                qualityvalueinput,
                genotypepathinput,
            )
            .unwrap();
            let command_logistic = variantclasslabel_logistic(
                fileinput,
                variantinput,
                qualityvalueinput,
                genotypepathinput,
            )
            .unwrap();
            let command_random = variantclasslabel_random(
                fileinput,
                variantinput,
                qualityvalueinput,
                genotypepathinput,
            )
            .unwrap();
            println!(
                "The commands have been finished:{}\n{}\n{}\n",
                command_knn, command_logistic, command_random
            );
        }
    }
}
