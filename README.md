# Handwritten-Numbers
Machine Learning project - *k* Nearest Neighbors for image processing
> If you use my code or a part of it for your projects, thank for quoting me!

I use the dataset MNIST784 from *scikit-learn* library which is 70 000 handwritten numbers between 0 and 9.
I recreate the dataset with images to be more flexible (with other images for other projects for instance).

You can choose a *k* and compute the optimal *k*.
You have all results (confusion matrix, different performances...) in a txt file.

Vocabulary I use for performances:

	accuracy
		ratio of right data for any value

	MCC (between -1 et +1)
		Matthews correlation coefficient; useful if classes have very different sizes

	prevalence
		ratio of a label among all labels

	precision
		ratio of real right elements among right elements

	recall
		ratio of right elements among right elements excepted

	F1-score (between 0 et 1)
		harmonic mean of precision and recall

	Fbeta-score (between 0 et 1)
		harmonic mean of precision and recall with the weight beta
