"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from azureml.core import Model, Run
import argparse


# Get workspace
run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace


parser = argparse.ArgumentParser("evaluate")
parser.add_argument(
    "--build_id",
    type=str,
    help="The ID of the build triggering this pipeline run",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the Model",
    default="azure_service_classifier",
)
args = parser.parse_args()

print("Argument 1: %s" % args.build_id)
print("Argument 2: %s" % args.model_name)
model_name = args.model_name
build_id = args.build_id

all_runs = exp.get_runs(include_children=True)

new_model_run = None

# get the last completed run with metrics
new_model_run = None
for run in all_runs:
    acc = run.get_metrics().get("val_accuracy")
    print(f'run is {run}, acc is {acc}')
    if run.get_status() == 'Finished' and acc is not None:
        new_model_run = run
        print('found a valid new model with acc {}'.format(acc))
        break

if new_model_run is None:
    raise Exception('new model must log a val_accuracy metric, please check'
                    ' your train.py file')
model_generator = Model.list(ws)

# Check that there are models
if len(model_generator) > 0:

    # Get the model with best val accuracy, assume this is best
    cur_max = None
    production_model = None

    for model in model_generator:
        cur_acc = model.run.get_metrics().get("val_accuracy")[-1]
        if cur_max is None or cur_acc > cur_max:
            cur_max = cur_acc
            production_model = model

    production_model_acc = cur_max
    new_model_acc = new_model_run.get_metrics().get(
        "val_accuracy")[-1]
    print(
        "Current Production model acc: {}, New trained model acc: {}".format(
            production_model_acc, new_model_acc
        )
    )

    promote_new_model = False
    if new_model_acc > production_model_acc or production_model_acc is None:
        promote_new_model = True
        print("New trained model performs better, will be registered")

else:
    promote_new_model = True
    print("This is the first model to be trained, \
          thus nothing to evaluate for now")

# Writing the run id to /aml_config/run_id.json
if promote_new_model:
    model_path = './outputs/exports'
    new_model_run.register_model(
        model_name=model_name,
        model_path=model_path,
        properties={"build_id": build_id, "run_type": "train"})
    print("Registered new model!")
