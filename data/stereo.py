import json
import string
from tqdm import tqdm

class StereoSet(object):
    def __init__(self, location, bias=None): 
        with open(location, "r") as f:
                self.json = json.load(f)
       
        self.version = self.json['version']
        self.intrasentence_examples = self.__create_intrasentence_examples__(
            self.json['data']['intrasentence'], bias)

    def __create_intrasentence_examples__(self, examples, bias):
        created_examples = []
        for example in examples:
            if bias is None or example['bias_type'] == bias:
                sentences = []
                for sentence in example['sentences']:
                    labels = []
                    for label in sentence['labels']:
                        labels.append(Label(**label))
                    sentence_obj = Sentence(
                        sentence['id'], sentence['sentence'], labels, sentence['gold_label'])
                    word_idx = None
                    for idx, word in enumerate(example['context'].split(" ")):
                        if "BLANK" in word: 
                            word_idx = idx
                    if word_idx is None:
                        raise Exception("No blank word found.")
                    template_word = sentence['sentence'].split(" ")[word_idx]
                    sentence_obj.template_word = template_word.translate(str.maketrans('', '', string.punctuation))
                    sentences.append(sentence_obj)
                created_example = IntrasentenceExample(
                    example['id'], example['bias_type'], 
                    example['target'], example['context'], sentences) 
                created_examples.append(created_example)
        return created_examples
    
    def get_intrasentence_examples(self):
        return self.intrasentence_examples

class Example(object):
    def __init__(self, ID, bias_type, target, context, sentences):
        """
         A generic example.

         Parameters
         ----------
         ID (string): Provides a unique ID for the example.
         bias_type (string): Provides a description of the type of bias that is 
             represented. It must be one of [RACE, RELIGION, GENDER, PROFESSION]. 
         target (string): Provides the word that is being stereotyped.
         context (string): Provides the context sentence, if exists,  that 
             sets up the stereotype. 
         sentences (list): a list of sentences that relate to the target. 
         """

        self.ID = ID
        self.bias_type = bias_type
        self.target = target
        self.context = context
        self.sentences = sentences

    def __str__(self):
        s = f"Domain: {self.bias_type} - Target: {self.target} \r\n"
        s += f"Context: {self.context} \r\n" 
        for sentence in self.sentences:
            s += f"{sentence} \r\n" 
        return s

class Sentence(object):
    def __init__(self, ID, sentence, labels, gold_label):
        """
        A generic sentence type that represents a sentence.

        Parameters
        ----------
        ID (string): Provides a unique ID for the sentence with respect to the example.
        sentence (string): The textual sentence.
        labels (list of Label objects): A list of human labels for the sentence. 
        gold_label (enum): The gold label associated with this sentence, 
            calculated by the argmax of the labels. This must be one of 
            [stereotype, anti-stereotype, unrelated, related].
        """

        assert type(ID)==str
        assert gold_label in ['stereotype', 'anti-stereotype', 'unrelated']
        assert isinstance(labels, list)
        assert isinstance(labels[0], Label)

        self.ID = ID
        self.sentence = sentence
        self.gold_label = gold_label
        self.labels = labels
        self.template_word = None

    def __str__(self):
        return f"{self.gold_label.capitalize()} Sentence: {self.sentence}"

class Label(object):
    def __init__(self, human_id, label):
        """
        Label, represents a label object for a particular sentence.

        Parameters
        ----------
        human_id (string): provides a unique ID for the human that labeled the sentence.
        label (enum): provides a label for the sentence. This must be one of 
            [stereotype, anti-stereotype, unrelated, related].
        """
        assert label in ['stereotype',
                         'anti-stereotype', 'unrelated', 'related']
        self.human_id = human_id
        self.label = label


class IntrasentenceExample(Example):
    def __init__(self, ID, bias_type, target, context, sentences):
        """
        Implements the Example class for an intrasentence example.

        See Example's docstring for more information.
        """
        super(IntrasentenceExample, self).__init__(
            ID, bias_type, target, context, sentences)
