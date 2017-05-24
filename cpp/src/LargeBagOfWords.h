#ifndef SRC_LARGEBAGOFWORDS_H_
#define SRC_LARGEBAGOFWORDS_H_

#include "opencv2/features2d.hpp"

namespace cv {
/*
 * Trainer for creating a bag-of-words vocabulary. This is specialized for use-
 * cases with very large vocabulary sizes (in the 100,000s). To that end, the
 * k-means algorithm (used for clustering) is adapted to use approximate
 * nearest neighbor lookups.
 *
 * This code is largely based on the OpenCV BOWKMeansTrainer.
 */
class CV_EXPORTS_W BOWApproxKMeansTrainer: public BOWKMeansTrainer {
public:
	CV_WRAP	BOWApproxKMeansTrainer(int clusterCount, const TermCriteria& termcrit =
			TermCriteria(), int attempts = 3, int flags = KMEANS_RANDOM_CENTERS);
	virtual ~BOWApproxKMeansTrainer();

	// Returns trained vocabulary (i.e. cluster centers).
	CV_WRAP virtual Mat cluster() const;
	CV_WRAP	virtual Mat cluster(const Mat& descriptors) const;
};

class CV_EXPORTS_W BOWSparseImgDescriptorExtractor : public BOWImgDescriptorExtractor{

public:
    CV_WRAP BOWSparseImgDescriptorExtractor( const Ptr<DescriptorExtractor>& dextractor,
                               const Ptr<DescriptorMatcher>& dmatcher );
    virtual ~BOWSparseImgDescriptorExtractor();


    /*
	 * Compute a bag of words descriptor, and return it as a sparse array. For
	 * a visual vocabulary of K words and images with N descriptors, this is
	 * efficient when N << K.
	 */
	void computeSparse(InputArray keypointDescriptors,
			SparseMat& imgDescriptorOut);
};

}

#endif /* SRC_LARGEBAGOFWORDS_H_ */
