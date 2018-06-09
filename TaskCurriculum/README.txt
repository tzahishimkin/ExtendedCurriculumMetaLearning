This is the Python implementation of the method described in the following paper: 
A. Pentina, V. Sharmanska, C. H. Lampert: Curriculum Learning of Multiple Tasks, CVPR 2015.
If you use or adapt this code, please, cite accordingly. 

=====Instructions=====

--- main_seq_learning.py contains our implementation of the SeqMT and MultiSeqMT algorithms 
for learning the tasks in a data-dependent order described in the paper. 
As a reference, we also provide a baseline when the tasks are learned in a pre-specified random order. 

--- Note, the trade-off parameter C is automatically tuned during inference (that is why it might be slow). 
The parameter range is specified in main_seq_learning.py, line 52.

--- To run the demo, simply type 'python run.py' in the terminal window. 
In our demo, we have a running example related to our second experiment in the paper using the Shoes dataset. 
The outcome of the program should be similar to the following:

	***Learning the tasks in a data-dependent order
	***SeqMT algorithm
	Avg. error rate across tasks: 11.10
	Learned order of tasks: ['T6' 'T5' 'T1' 'T10' 'T2' 'T8' 'T7' 'T9' 'T4' 'T3']
	***MultiSeqMT algorithm
	Avg. error rate across tasks: 9.30
	***As a reference
	***Learning the tasks in a random order
	Avg. error rate across tasks: 13.70
	Random order of tasks: ['T4' 'T2' 'T5' 'T1' 'T6' 'T8' 'T7' 'T9' 'T10' 'T3']

 


