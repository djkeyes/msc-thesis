
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "CustomSiftKeypoints.h"

using std::vector;
using std::cout;
using std::endl;
using namespace cv;

namespace sdl {

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// intermediate type used for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;


static Mat createInitialImage( const Mat& img, bool doubleImageSize, float sigma )
{
    Mat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
    }
    else
        img.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);

    float sig_diff;

    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        Mat dbl;
        resize(gray_fpt, dbl, Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, INTER_LINEAR);
        GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
        return dbl;
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
        return gray_fpt;
    }
}

void buildGaussianPyramid( const Mat& base, vector<Mat>& pyr, int nOctaves, int nOctaveLayers, double sigma )
{
    vector<double> sig(nOctaveLayers + 3);
    pyr.resize(nOctaves*(nOctaveLayers + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = std::pow( 2., 1. / nOctaveLayers );
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = std::pow(k, (double)(i-1))*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 3; i++ )
        {
            Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                const Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
                resize(src, dst, Size(src.cols/2, src.rows/2),
                       0, 0, INTER_NEAREST);
            }
            else
            {
                const Mat& src = pyr[o*(nOctaveLayers + 3) + i-1];
                GaussianBlur(src, dst, Size(), sig[i], sig[i]);
            }
        }
    }
}

class buildDoGPyramidComputer : public ParallelLoopBody
{
public:
    buildDoGPyramidComputer(
        int _n_octave_layers,
        const vector<Mat>& _gpyr,
        vector<Mat>& _dogpyr)
        : n_octave_layers(_n_octave_layers),
          gpyr(_gpyr),
          dogpyr(_dogpyr) { }

    void operator()( const cv::Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;

        for( int a = begin; a < end; a++ )
        {
            const int o = a / (n_octave_layers + 2);
            const int i = a % (n_octave_layers + 2);

            const Mat& src1 = gpyr[o*(n_octave_layers + 3) + i];
            const Mat& src2 = gpyr[o*(n_octave_layers + 3) + i + 1];
            Mat& dst = dogpyr[o*(n_octave_layers + 2) + i];
            subtract(src2, src1, dst, noArray(), DataType<sift_wt>::type);
        }
    }

private:
    int n_octave_layers;
    const vector<Mat>& gpyr;
    vector<Mat>& dogpyr;
};


void buildDoGPyramid( const vector<Mat>& gpyr, vector<Mat>& dogpyr, int n_octave_layers )
{
    int nOctaves = (int)gpyr.size()/(n_octave_layers + 3);
    dogpyr.resize( nOctaves*(n_octave_layers + 2) );

    parallel_for_(Range(0, nOctaves * (n_octave_layers + 2)), buildDoGPyramidComputer(n_octave_layers, gpyr, dogpyr));
}

// Computes a gradient orientation histogram at a specified pixel
static float calcOrientationHist( const Mat& img, Point pt, int radius,
                                  float sigma, float* hist, int n )
{
    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    AutoBuffer<float> buf(len*4 + n+4);
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    k = 0;
#if CV_AVX2
    if( USE_AVX2 )
    {
        __m256 __nd360 = _mm256_set1_ps(n/360.f);
        __m256i __n = _mm256_set1_epi32(n);
        int CV_DECL_ALIGNED(32) bin_buf[8];
        float CV_DECL_ALIGNED(32) w_mul_mag_buf[8];
        for ( ; k <= len - 8; k+=8 )
        {
            __m256i __bin = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(__nd360, _mm256_loadu_ps(&Ori[k])), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));

            __bin = _mm256_sub_epi32(__bin, _mm256_andnot_si256(_mm256_cmpgt_epi32(__n, __bin), __n));
            __bin = _mm256_add_epi32(__bin, _mm256_and_si256(__n, _mm256_cmpgt_epi32(_mm256_setzero_si256(), __bin)));

            __m256 __w_mul_mag = _mm256_mul_ps(_mm256_loadu_ps(&W[k]), _mm256_loadu_ps(&Mag[k]));

            _mm256_store_si256((__m256i *) bin_buf, __bin);
            _mm256_store_ps(w_mul_mag_buf, __w_mul_mag);

            temphist[bin_buf[0]] += w_mul_mag_buf[0];
            temphist[bin_buf[1]] += w_mul_mag_buf[1];
            temphist[bin_buf[2]] += w_mul_mag_buf[2];
            temphist[bin_buf[3]] += w_mul_mag_buf[3];
            temphist[bin_buf[4]] += w_mul_mag_buf[4];
            temphist[bin_buf[5]] += w_mul_mag_buf[5];
            temphist[bin_buf[6]] += w_mul_mag_buf[6];
            temphist[bin_buf[7]] += w_mul_mag_buf[7];
        }
    }
#endif
    for( ; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    // smooth the histogram
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];

    i = 0;
#if CV_AVX2
    if( USE_AVX2 )
    {
        __m256 __d_1_16 = _mm256_set1_ps(1.f/16.f);
        __m256 __d_4_16 = _mm256_set1_ps(4.f/16.f);
        __m256 __d_6_16 = _mm256_set1_ps(6.f/16.f);
        for( ; i <= n - 8; i+=8 )
        {
#if CV_FMA3
            __m256 __hist = _mm256_fmadd_ps(
                _mm256_add_ps(_mm256_loadu_ps(&temphist[i-2]), _mm256_loadu_ps(&temphist[i+2])),
                __d_1_16,
                _mm256_fmadd_ps(
                    _mm256_add_ps(_mm256_loadu_ps(&temphist[i-1]), _mm256_loadu_ps(&temphist[i+1])),
                    __d_4_16,
                    _mm256_mul_ps(_mm256_loadu_ps(&temphist[i]), __d_6_16)));
#else
            __m256 __hist = _mm256_add_ps(
                _mm256_mul_ps(
                        _mm256_add_ps(_mm256_loadu_ps(&temphist[i-2]), _mm256_loadu_ps(&temphist[i+2])),
                        __d_1_16),
                _mm256_add_ps(
                    _mm256_mul_ps(
                        _mm256_add_ps(_mm256_loadu_ps(&temphist[i-1]), _mm256_loadu_ps(&temphist[i+1])),
                        __d_4_16),
                    _mm256_mul_ps(_mm256_loadu_ps(&temphist[i]), __d_6_16)));
#endif
            _mm256_storeu_ps(&hist[i], __hist);
        }
    }
#endif
    for( ; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval;
}


//
// Detects features at extrema in DoG scale space.
void findScaleSpaceExtrema( const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
                                  vector<KeyPoint>& keypoints, int n_octave_layers, double sigma )
{
    int nOctaves = (int)gauss_pyr.size()/(n_octave_layers + 3);
    const int n = SIFT_ORI_HIST_BINS;
    float hist[n];
    KeyPoint kpt;

    vector<KeyPoint> keypoints_out;

    for( int o = 0; o < nOctaves; o++ ) {
        for( int i = 1; i <= n_octave_layers; i++ )
        {
            int idx = o*(n_octave_layers+2)+i;
            const Mat& img = dog_pyr[idx];
            const Mat& prev = dog_pyr[idx-1];
            const Mat& next = dog_pyr[idx+1];

            for (auto& input_kpt : keypoints) {
            	int x = static_cast<int>(input_kpt.pt.x);
            	int y = static_cast<int>(input_kpt.pt.y);

            	int c = x / (1 << o);
            	int r = y / (1 << o);

            	const sift_wt* currptr = img.ptr<sift_wt>(r);
                const sift_wt* prevptr = prev.ptr<sift_wt>(r);
                const sift_wt* nextptr = next.ptr<sift_wt>(r);

				sift_wt val = currptr[c];

				cout << "o=" << o << ", i=" << i << ", idx=" << idx << ", nOctaves=" << nOctaves << ", n_octave_layers="
						<< n_octave_layers << ", r=" << r << ", c=" << c << ", prevsize=" << prev.size() << ", cursize="
						<< img.size() << ", nextsize=" << next.size() << endl;
				cout << "cur: " << img.at<sift_wt>(r, c) << endl;
				cout << "prev: " << prev.at<sift_wt>(r, c) << endl;
				cout << "next: " << next.at<sift_wt>(r, c) << endl;
				float foo;
				foo = nextptr[c];
				cout << foo << endl;
				foo = prevptr[c];
				cout << foo << endl;
				// find local extrema (strictly in scale space) with strictly pixel accuracy
				if( (val > 0 && val >= nextptr[c] && val >= prevptr[c]) ||
					(val < 0 && val <= nextptr[c] && val <= prevptr[c] ))
				{
					int r1 = r, c1 = c, layer = i;
					float scl_octv = kpt.size*0.5f/(1 << o);
					float omax = calcOrientationHist(gauss_pyr[o*(n_octave_layers+3) + layer],
													 Point(c1, r1),
													 cvRound(SIFT_ORI_RADIUS * scl_octv),
													 SIFT_ORI_SIG_FCTR * scl_octv,
													 hist, n);
					float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
					for( int j = 0; j < n; j++ )
					{
						int l = j > 0 ? j - 1 : n - 1;
						int r2 = j < n-1 ? j + 1 : 0;

						if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
						{
							KeyPoint new_kpt;
							new_kpt.pt.x = c * (1 << o);
							new_kpt.pt.y = r * (1 << o);
							new_kpt.octave = o + (layer << 8) + (cvRound(0.5*255) << 16);
							new_kpt.size = sigma*powf(2.f, static_cast<float>(layer) / n_octave_layers)*(1 << o)*2;

							float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
							bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
							new_kpt.angle = 360.f - (float)((360.f/n) * bin);
							if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
								kpt.angle = 0.f;
							keypoints_out.push_back(new_kpt);
						}
					}
				}
            }
        }
    }

    keypoints = keypoints_out;
}


void computeSiftOrientationAndScale(Mat image, vector<KeyPoint>& keypoints, int num_octave_layers, double sigma) {

	int first_octave = -1;

    vector<Mat> gpyr, dogpyr;
    Mat base = createInitialImage(image, true, (float)sigma);

    int num_octaves = cvRound(std::log( (double)std::min( base.cols, base.rows ) ) / std::log(2.) - 2) - first_octave;

	buildGaussianPyramid(base, gpyr, num_octaves, num_octave_layers, sigma);
	buildDoGPyramid(gpyr, dogpyr, num_octave_layers);

	findScaleSpaceExtrema(gpyr, dogpyr, keypoints, num_octaves, sigma);
	KeyPointsFilter::removeDuplicated( keypoints );

	for( size_t i = 0; i < keypoints.size(); i++ )
	{
		KeyPoint& kpt = keypoints[i];
		float scale = 1.f/(float)(1 << -first_octave);
		kpt.octave = (kpt.octave & ~255) | ((kpt.octave + first_octave) & 255);
		kpt.pt *= scale;
		kpt.size *= scale;
	}
}

}
