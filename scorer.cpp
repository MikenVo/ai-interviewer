#include <iostream>
#include <vector>
#include <algorithm>

// This module simulates the high-performance scoring engine required by the prompt.
// It would be compiled and called by the Python backend in a production environment.

struct InterviewData {
	int keywords_matched;
	double response_time_avg;
	int code_complexity_score;
	bool cv_has_required_elements;
};

struct FinalScore {
	double cv_score;
	double knowledge_score;
	double attitude_score;
	double thinking_score;
	bool is_hired;
};

class AIScorer {
public:
	FinalScore calculate(InterviewData data) {
		FinalScore score;

		// CV Scoring Algorithm
		score.cv_score = data.cv_has_required_elements ?
			std::min(10.0, 5.0 + (data.keywords_matched * 0.2)) : 0.0;

		// Knowledge Scoring
		score.knowledge_score = std::min(10.0, (double)data.keywords_matched * 0.8);

		// Attitude Scoring (Inverse to response time usually implies confidence, simplified logic)
		score.attitude_score = data.response_time_avg < 60.0 ? 9.0 : 7.0;

		// Thinking/Code Scoring
		score.thinking_score = std::min(10.0, (double)data.code_complexity_score * 2.0);

		// Hiring Decision
		double total_avg = (score.cv_score + score.knowledge_score + score.attitude_score + score.thinking_score) / 4.0;
		score.is_hired = total_avg >= 7.0 && score.cv_score > 0;

		return score;
	}
};

extern "C" {
	// Interface for Python CTypes
	void grade_candidate(int k, double t, int c, bool cv, double* results) {
		AIScorer scorer;
		InterviewData data = { k, t, c, cv };
		FinalScore fs = scorer.calculate(data);

		results[0] = fs.cv_score;
		results[1] = fs.knowledge_score;
		results[2] = fs.attitude_score;
		results[3] = fs.thinking_score;
		results[4] = fs.is_hired ? 1.0 : 0.0;
	}
}