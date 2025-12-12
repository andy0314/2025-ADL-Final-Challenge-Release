import os
import json

class Target():
    '''
    Temporary implementation of Target class
    '''
    def __init__(self, model):
        self.model = model

    def respond(self, prompt):
        response = self.model.generate(
            "You are a helpful assistant.",
            prompt,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        return response
