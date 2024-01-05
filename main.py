from helpers import Timer
from q24 import q24
from q25 import q25


def question_wrapper_factory(title):
    def question_wrapper(q_answer_fn):
        print('------------')
        with Timer():
            answer = q_answer_fn()
            print(f"Answer(s) to {title}: {' and '.join([str(part) for part in answer if part is not None])}")
        print('------------')
    return question_wrapper


question_wrapper_factory('Q24')(q24)
question_wrapper_factory('Q25')(q25)