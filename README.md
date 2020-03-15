# AIAP Technical Assessment

Design and create a simple machine learning pipeline that will ingest/process the entailed
dataset and feed it into the machine learning algorithm(s) of your choice, returning metric outputs. 

## Explanation of mlp.py
This is an end to end machine learning pipeline that considers 5 regression models:
1. Linear Regression 
2. Support Vector Regression
3. Decision Tree Regression
4. Random Forest Regression
5. XGBoost Regression

The reason for choosing the 5 models is because the dependent variable is continuous, 
and the distribution of the dependent variable is non-Gaussian.

Data was scaled using MinMaxScaler as StandardScaler would remove the unique feature of the dataset,
which is that it has a long right tail.

### Prerequisites

You will need to install python 3.6.7, and install all packages in their respective versions by running this in the command line

```
pip3 install -r requirements.txt
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
