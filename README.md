---
page_type: sample
languages:
- python
products:
- azureml
description: "Add 150 character max description"
urlFragment: "update-this-to-unique-url-stub"
---
**TODO** - add a build badge here
# Automated StackOverflow question tagging with BERT and TensorFlow 2.0

<!-- 
Guidelines on README format: https://review.docs.microsoft.com/help/onboard/admin/samples/concepts/readme-template?branch=master

Guidance on onboarding samples to docs.microsoft.com/samples: https://review.docs.microsoft.com/help/onboard/admin/samples/process/onboarding?branch=master

Taxonomies for products and languages: https://review.docs.microsoft.com/new-hope/information-architecture/metadata/taxonomies?branch=master
-->

Use this repository to learn how to fine-tune a BERT model to tag StackOverflow questions automatically with Azure Machine Learning.

MLOps build status: [![Build Status](https://dev.azure.com/gopalv0408/tfworld_test/_apis/build/status/microsoft-ci-build-official?branchName=master)](https://dev.azure.com/gopalv0408/tfworld_test/_build/latest?definitionId=17&branchName=master)

## Contents

Check out the file contents below.

| File/folder       | Description                                |
|-------------------|--------------------------------------------|
| `ml_ops`          | Directories and files for the MLOps portion of the demo                      |
| `data`            | Stores the data used in the interpretability section (other data comes from blob storage)                 |
| `scripts`         | Stores some utility scripts used for data preparation                  |
| `batch`           | Batch scoring assets                 |
| `.gitignore`      | Define what to ignore at commit time.      |
| `README.md`       | This README file.                          |
| `LICENSE`         | The license for the sample.                |

## Prerequisites

If you run the included notebooks Azure notebook, you will need to `pip install` the `transformers` library.

## Setup

Create a Notebook VM by logging onto the Azure ML [workspace](https://ml.azure.com/) and creating a notebook. Clone this repository to your VM in order to run this code.

## Running the sample

**TODO**
Outline step-by-step instructions to execute the sample and see its output. Include steps for executing the sample from the IDE, starting specific services in the Azure portal or anything related to the overall launch of the code.

## Key concepts

**TODO**
Provide users with more context on the tools and services used in the sample. Explain some of the code that is being used and how services interact with each other.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
