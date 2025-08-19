import dspy
import random
from dspy.teleprompt import Teleprompter, LabeledFewShot

class BanditTeleprompter(Teleprompter):
    def __init__(self, metric, k=3, num_candidates=10):
        self.metric = metric
        self.k = k
        self.num_candidates = num_candidates

    def compile(self, student, *, trainset):
        best_program = None
        best_score = -1

        for i in range(self.num_candidates):
            # Randomly select k examples from the training set
            candidate_examples = random.sample(trainset, self.k)

            # Use LabeledFewShot to create a candidate program with the selected examples
            optimizer = LabeledFewShot(k=self.k)
            candidate_program = optimizer.compile(student, trainset=candidate_examples)

            # Evaluate the candidate program on the full training set
            total_score = 0
            for dev_example in trainset:
                prediction = candidate_program(**dev_example.inputs())
                metric_result = self.metric(dev_example, prediction)
                total_score += metric_result["score"]
            
            score = total_score / len(trainset)

            if score > best_score:
                best_score = score
                best_program = candidate_program

        return best_program