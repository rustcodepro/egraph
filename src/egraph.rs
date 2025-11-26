use ndarray::Array1;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::ndarray;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::accuracy;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
codeprog@icloud.com

- a streamline machine learning crate to how to use the population variant
  data from the eVai or the other variants for the machine learning and predicts
  and confirm where the variant data is not annotated.
  see the test files as how to prepare the data for the vairant classification.
*/

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct AutoencoderData {
    pub nameallele: String,
    pub gentoype: f64,
    pub quality: f64,
}

/*
Taking the variant and making the classification pass and
here i have all the variants that have been annotated and passed.
*/

type EGRAPHTYPE = Vec<AutoencoderData>;
pub fn readdata(path: &str) -> Result<EGRAPHTYPE, Box<dyn std::error::Error>> {
    let fileopen = File::open(path).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut tensorvec: Vec<AutoencoderData> = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("file not present");
        let linevec = line.split("\t").collect::<Vec<_>>();
        if linevec[55] == "pass" {
            tensorvec.push(AutoencoderData {
                nameallele: linevec[4].to_string(),
                gentoype: linevec[53].parse::<f64>().unwrap(),
                quality: linevec[54].parse::<f64>().unwrap(),
            })
        }
    }
    Ok(tensorvec)
}

pub fn logisticclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdata(pathfile).unwrap();
    let vecgenotype: Vec<&Vec<f64>> = Vec::new();
    let classlabels: Vec<f64> = Vec::new();
    for i in inputvariable.iter() {
        if i.quality < quality.parse::<f64>().unwrap() {
            classlabels.push(0 as f64);
        } else if i.quality > quality.parse::<f64>().unwrap() {
            classlabels.push(1 as f64);
        }
        vecgenotype.push(&vec![i.gentoype as f64]);
    }
    let features_2d: Array2<f64> = Array2::from_shape_vec((1, 1), vecgenotype);
    let densematrix: Result<DenseMatrix<f64>, smartcore::error::Failed> =
        DenseMatrix::from_2d_array(features_2d);
    let model = LogisticRegression::fit(&features_2d, &classlabels, Default::default()).unwrap();
    println!("The logistic model has been predicted:{:?}", model);
    let prediction = predictiondata(genotypespath).unwrap();
    let predictions_value = model.predict(&prediction);
    let accuracypred = accuracy(&prediction, &predictions_value);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    writeln!(
        filewrite,
        "The model predicted accuracy is: {}",
        accuracypred
    );
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The logistic model has finished with the accuracy".to_string())
}

pub fn knnclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdata(pathfile).unwrap();
    let vecgenotype: Vec<f64> = Vec::new();
    let classlabels: Vec<f64> = Vec::new();
    for i in inputvariable.iter() {
        if i.quality < quality.parse::<f64>().unwrap() {
            classlabels.push(0 as f64);
        } else if i.quality > quality.parse::<f64>().unwrap() {
            classlabels.push(1 as f64);
        }
        vecgenotype.push(i.gentoype as f64);
    }
    let features_2d: Array2<f64> = Array2::from_shape_vec((1, 1), vecgenotype);
    let densematrix: Result<DenseMatrix<f64>, smartcore::error::Failed> =
        DenseMatrix::from_2d_array(features_2d).unwrap();
    let model = KNNClassifier::fit(&features_2d, &classlabels, Default::default()).unwrap();
    println!("The logistic model has been predicted:{:?}", model);
    let prediction = predictiondata(genotypespath).unwrap();
    let predictions_value = model.predict(prediction);
    let accuracypred = accuracy(prediction, predictions_value);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    for i in predictions_value.into_iter() {
        writeln!(
            filewrite,
            "The model has predicted the value with the accuracy:{}",
            accuracypred
        );
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The knn classification model has finished with the accuracy".to_string())
}

pub fn randomforestclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let inputvariable = readdata(pathfile).unwrap();
    let vecgenotype: Vec<f64> = Vec::new();
    let classlabels: Vec<f64> = Vec::new();
    for i in inputvariable.iter() {
        if i.quality < quality.parse::<f64>().unwrap() {
            classlabels.push(0 as f64);
        } else if i.quality > quality.parse::<f64>().unwrap() {
            classlabels.push(1 as f64);
        }
        vecgenotype.push(i.gentoype as f64);
    }
    let features_2d: Array2<f64> = Array2::from_shape_vec((1, 1), vecgenotype);
    let densematrix: DenseMatrix<64> = DenseMatrix::from_2d_array(features_2d);
    let model =
        RandomForestClassifier::fit(&features_2d, &classlabels, Default::default()).unwrap();
    println!("The logistic model has been predicted:{:?}", model);
    let prediction = predictiondata(genotypespath).unwrap();
    let predictions_value = model.predict(prediction);
    let filewrite = File::open("predictedvalue.txt").expect("The file not present");
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The random forest model has finished with the accuracy".to_string())
}

pub fn predictiondata(pathfile: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut record: Vec<f64> = Vec::new();
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    for file in fileread.lines() {
        let fileline = file.expect("line not present");
        record.push(fileline.parse::<f64>().unwrap());
    }
    Ok(record)
}
pub fn variantclasslabel_logistic(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let inputvariable = readdata(pathfile).unwrap();
    let vecgenotype: Vec<&Vec<f64>> = Vec::new();
    let classlabels: Vec<f64> = Vec::new();
    for i in inputvariable.iter() {
        if i.nameallele == "variant" && i.quality < quality.parse::<f64>().unwrap() {
            classlabels.push(0 as f64);
        } else if i.quality > quality.parse::<f64>().unwrap() {
            classlabels.push(1 as f64);
        }
        vecgenotype.push(&vec![i.gentoype as f64, i.quality as f64]);
    }
    let features_2d: Array2<f64> = Array2::from_shape_vec((2, 1), &vecgenotype);
    let densematrix: DenseMatrix<64> = DenseMatrix::from_2d_array(features_2d);
    let model = LogisticRegression::fit(&features_2d, &classlabels, Default::default()).unwrap();
    println!("The logistic model has been predicted:{:?}", model);
    let prediction = predictiondata(&genotypespath).unwrap();
    let predictions_value = model.predict(&prediction);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok(
        "The random forest model with the filtered variant has finished with the accuracy"
            .to_string(),
    )
}

pub fn variantclasslabel_knn(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let inputvariable = readdata(pathfile).unwrap();
    let vecgenotype: Vec<&Vec<f64>> = Vec::new();
    let classlabels: Vec<f64> = Vec::new();
    for i in inputvariable.iter() {
        if i.nameallele == "variant" && i.quality < quality.parse::<f64>().unwrap() {
            classlabels.push(0 as f64);
        } else if i.quality > quality.parse::<f64>().unwrap() {
            classlabels.push(1 as f64);
        }
        vecgenotype.push(&vec![i.gentoype as f64, i.quality as f64]);
    }
    let features_2d: Array2<f64> = Array2::from_shape_vec((2, 1), &vecgenotype);
    let densematrix: DenseMatrix<64> = DenseMatrix::from_2d_array(features_2d);
    let model = KNNClassifier::fit(&features_2d, &classlabels, Default::default()).unwrap();
    println!("The logistic model has been predicted:{:?}", model);
    let prediction = predictiondata(&genotypespath).unwrap();
    let predictions_value = model.predict(&prediction);
    let filewrite = File::open("predictedvalue.txt").expect("The file not present");
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok(
        "The random forest model with the filtered variant has finished with the accuracy"
            .to_string(),
    )
}

pub fn variantclasslabel_random(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let inputvariable = readdata(pathfile).unwrap();
    let vecgenotype: Vec<&Vec<f64>> = Vec::new();
    let classlabels: Vec<f64> = Vec::new();
    for i in inputvariable.iter() {
        if i.nameallele == "variant" && i.quality < quality.parse::<f64>().unwrap() {
            classlabels.push(0 as f64);
        } else if i.quality > quality.parse::<f64>().unwrap() {
            classlabels.push(1 as f64);
        }
        vecgenotype.push(&vec![i.gentoype as f64, i.quality as f64]);
    }
    let features_2d: Array2<f64> = Array2::from_shape_vec((2, 1), &vecgenotype);
    let densematrix: f<64> = DenseMatrix::from_2d_array(features_2d);
    let model =
        RandomForestClassifier::fit(&features_2d, &classlabels, Default::default()).unwrap();
    println!("The logistic model has been predicted:{:?}", model);
    let prediction = predictiondata(&genotypespath).unwrap();
    let predictions_value = model.predict(&prediction);
    let filewrite = File::open("predictedvalue.txt").expect("The file not present");
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok(
        "The random forest model with the filtered variant has finished with the accuracy"
            .to_string(),
    )
}
