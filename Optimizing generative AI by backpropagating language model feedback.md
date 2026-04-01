

| Article |  |  |
| :---- | :---- | ----- |
| **Optimizing generative AI by backpropagating language model feedback** |  |  |
| [https://doi.org/10.1038/s41586-025-08661-4](https://doi.org/10.1038/s41586-025-08661-4)  Received: 12 June 2024 Accepted: 16 January 2025 Published online: 19 March 2025 [![][image1]](http://crossmark.crossref.org/dialog/?doi=10.1038/s41586-025-08661-4&domain=pdf)Check for updates | **MertYuksekgonul1,4**✉**, Federico Bianchi1,4, Joseph Boen2,4, Sheng Liu2,4, Pan Lu2,4, Zhi Huang2,4, Carlos Guestrin1,3 & James Zou1,2,3** ✉  Recent breakthroughsin artificial intelligence (AI) are increasingly driven by systems orchestrating multiple large language models (LLMs) and other specialized tools, such as search engines and simulators. So far, these systems are primarily handcrafted by domain experts and tweaked through heuristics rather than being automatically optimized, presenting a substantial challenge to accelerating progress. The development of artificial neural networks faceda similar challenge until backpropagation and automatic diferentiation transformed the field by making optimization turnkey. Analogously, here we introduce TextGrad, a versatile framework that performs optimization by backpropagating LLM-generated feedback to improve AI systems. By leveraging natural language feedback to critique and suggest improvements to any partofa system—from prompts to outputs such as molecules or treatment plans—TextGrad enables the automatic optimization of generative AI systems across diverse tasks. We demonstrate TextGrad’s generality andefectiveness through studies in solving PhD-level science problems, optimizing plans for radiotherapy treatments, designing molecules with specific properties, coding, and optimizing agentic systems. TextGrad empowers scientists and engineers to easily develop impactful generative AI systems. |  |
| Large language models (LLMs) are transforming how breakthrough |  | To address this challenge, we introduce TextGrad to perform auto- |
| artificial intelligence (AI) systems are built[1](#bookmark=id.64o75th0a9mz). The new generation of |  | matic ‘differentiation’via text. While neural networks communicate |
| AI applications increasingly operate as compound systems, orches- |  | through numerical values, modern AI systems communicate through |
| trating multiple sophisticated components—from LLM-based agents |  | unstructured data such as text, code, and images. In TextGrad, each AI |
| to specialized tools such as simulators and web search engines. For |  | system is transformed into a computation graph where components |
| instance, a system of LLMs communicating with symbolic solvers can |  | exchange these rich, unstructured variables through complex (not |
| solve olympiad-level maths problems[2](#bookmark=id.64o75th0a9mz); and a system of LLMs using |  | necessarily differentiable) functions. We use‘differentiation’asa meta- |
| search engines and code interpreter tools performs comparably to |  | phor, where LLMs provide feedback (dubbed‘textual gradients’[15](#bookmark=id.64o75th0a9mz)) in |
| human competitive programmers[3](#bookmark=id.64o75th0a9mz)  and solves real-world GitHub |  | the form of informative and interpretable natural language critiques, |
| issues[4](#bookmark=id.64o75th0a9mz). Despite these advances, many of these breakthroughs came |  | describing how each variable should be modified to improve the overall |
| from systems that are handcrafted by domain experts. Therefore, |  | system. Textual gradients are propagated through arbitrary functions, |
| developing automated optimization algorithms is one of the most |  | such asLLM application-programming-interface (API) calls, simulators |
| crucial challenges to build compound systems with LLMsand acceler- |  | or external numerical solvers. Notably, the text feedback does not |
| ate future breakthroughs[5](#bookmark=id.64o75th0a9mz) –[7](#bookmark=id.64o75th0a9mz). |  | target the parameters of the underlying neural networks, and is thus |
| For the past 15 years, many advances inAI have relied on neural |  | compatible with black-box APIs. |
| networks and differentiable optimization[8](#bookmark=id.64o75th0a9mz) –[12](#bookmark=id.64o75th0a9mz). Components of neu- |  | We demonstrate the power of TextGradin diverse domains, ranging |
| ral networks (for example, two artificial neurons) communicate |  | from question-answering benchmarksto radiotherapy treatment plan |
| through differentiable functions such as matrix multiplications[13](#bookmark=id.64o75th0a9mz) . |  | optimization and molecule generation (Fig. [1](#bookmark=id.64o75th0a9mz)), all with a single frame- |
| Therefore, communicating numerical gradients through each com- |  | work without modifications for different tasks. LLMs can provide rich, |
| ponent by backpropagation[14](#bookmark=id.64o75th0a9mz), which provides the direction to adjust |  | legible and expressive natural language feedback to variables in this |
| each parameter to improve a model, has been the natural way to train |  | wide range of domains, such as proposing modifications to molecules, |
| AI models. However, differential optimization does not work easily |  | prompts to other LLMsand code snippets. Our framework is built on |
| for the new generation of generative AI systems because these sys- |  | the premise that current state-of-the-art LLMs can reason about indi- |
| tems often involve natural language interactions, black-box LLMs |  | vidual components and subtasks of the system that they optimize. Our |
| or external tools, making backpropagating of numerical gradients |  | results demonstrate the promise of TextGrad to automatically improve |
| difficult. |  | generative AI systems and its outputs. |
| 1Department of Computer Science, Stanford University, Stanford, CA, USA. 2Department of Biomedical Data Science, Stanford University, Stanford, CA, USA. 3Chan Zuckerberg Biohub, San Francisco, CA, USA. 4These authors contributed equally: Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang. ✉e-mail: [merty@stanford.edu](mailto:merty@stanford.edu); [jamesz@stanford.edu](mailto:jamesz@stanford.edu) |  |  |

Nature  | Vol 639  | 20 March 2025  | **609**

**Article**

“this prompt can

**a**

*∂*Loss

**b**

*∂*Loss

\= 0.124

“this response can be improved by...”

\=

*∂*Loss

*∂*Prompt

be improved by...”

*∂*o1                   *h*1

Prompt

\=

o1

Output

*∂*Response

*∂*Loss *∂h*1

\= 0.267

0.3

0.1

1.2  
![][image2]

![][image3]    LLM  
![][image4]

–0.2

0.3

Final output

Output

Query

0.2

0.7

0.5

0.7

0.5

![][image5] Search engine

![][image6] LLM

![][image7] LLM

0.5

![][image8]

–0.1

–0.05

–0.4

–0.1

![][image9]  Tool caller

Input layer              Hidden layer

–0.1

Output

Tool prompt

Output

Loss

Intermediate outputs

Inputs

![][image10]                        Loss Evaluation instruction

**c**  
1\.  Analogy in abstractions

| Maths | PyTorch | ![][image11] TextGrad |
| :---- | :---- | :---- |

Input Model

z                          Tensor(image)                                     tg . variable(article)

y ![][image12] fo(ar)                   ResNet50()                                          tg . BlackboxLLM( ![][image13] You  are  a  summarize r . " )

Loss Optimizer

Lty, y) ![][image14]i log(yi)

![][image15]L  i         ![][image16]L  
GD(9, —) \=          

A

crossEntropyLoss()

SGD(list(model . parameters()))

tg . TextLoss( " Rate  the  summary . " ) tg . TGD(List(model . parameters()))

2\.   Automatic differentiation

PyTorch and TextGrad share the same

Forward pass                                Backward pass                 Updating variable

syntax for backpropagation and optimization

loss  \=  loss  fn(model(input))     loss . backward()        optimizer . step()

**e**

**d**

| Gradients |
| :---- |
| Add functional groups that increase polarity for stronger interactions |

Molecule at iteration 1  Vina score: –4.3 kcal mol–1

O

| Gradients |
| :---- |
| \*\*Handling \`nums\[i\] \== k\`\*\*: The current logic does not correctly handle the case when \`nums\[i\] \== k\`. The balance should be reset or adjusted differently when \`k\` is encountered. ... |

| for | i  in  range(n) :  if  nums \[i\]   \<  k : balance  \- \=  1 elif  nums \[i\]  \>  k : balance  \+=  1 if  nums \[i\]  \==  k : result  \+=  count . get(balance count . get(balance  \-  1 , |   ![][image17] |  \+ |
| :---- | :---- | :---- | :---- |
| else : result  \+=  count . get(balance ,   0\) count![][image18]balance\]   \=  count . get(balance , |  |  | ![][image19] |

Code at iteration *t*

HO

Molecule at iteration 2  Vina score: –5.5 kcal mol–1

| Gradients |
| :---- |
| Introduce polar or aromatic groups to increase interaction points |

O

N  
H

Molecule at iteration 3  Vina score: –7.5 kcal mol–1

| for  i  in  range(n) :  if  nums \[i\]  \<  k : balance  \- \=  1 elif  nums \[i\]   \>  k : balance  \+=  1 |  |  |  |  |  |  |  |  |
| ----- | ----- | :---- | :---- | ----- | :---- | :---- | :---- | :---- |
| el if | se : fou nums | nd\_\_k   \= \[i\]   \== | True k : |  |  |  |  |  |
|  | res | ult  \+= | count . | get(balance ,   0 |  | \+ |  |  |
| el | se : COU | count nt\[bal | · get(ba ance\]  \= | lance  \-   1 ,   0\) count . get(bal | ance , |  | ![][image20] |  |

Code at iteration *t* \+ 1

**g**

**f**

High dosage in bladder and  
rectum

| Gradients |
| :---- |
| The current weight   for the rectum and  bladder are relatively low, which is not sufﬁcient to protect the rectum and bladder... |

| You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value. |
| :---- |

Prompt at initialization (accuracy \= 77.8%)

Better protected bladder and  
rectum

Low ![][image21] High

Dose

| You will answer a reasoning question. List each item and its quantity in a clear and consistent format, such as '- Item: Quantity'. Sum the values directly from the list and provide a concise summation. Ensure the ﬁnal answer is clearly indicated in the format: 'Answer: $VALUE' where VALUE is a numerical value. Verify the relevance of each item to the context of the query and handle potential errors or ambiguities in the input. Double-check the ﬁnal count to ensure accuracy." |
| :---- |

Prompt after optimization (accuracy \= 91.9%)

**Fig. 1 | Overview of TextGrad. a**, Neural network and backpropagation using  numerical gradients. Backpropagation of gradients is the driving force of deep learning. Numerical gradientsofa loss function with respect to variables such as *ν*1 are backpropagated through hidden states such as*h*1. **b**, Black-box AI  
systems and backpropagation using natural language‘gradients’. We do not have gradients for compound systems of black-box AI systems, but we can   construct analogous backpropagation for text-based feedback, forming the basis of TextGrad. **c**, TextGrad shares the same abstractionsand syntax as

PyTorch, making it generalizable and easy to learn. In both cases, we use the same abstractions to instantiate network variables *θ*, target and predicted

outputs*y* and , model*f*, loss function *L*, and gradient descent optimizer GD.

For instance, in TextGrad, a model*f* can bea blackbox language model, where we consider aprompt tobe the parameter we want to optimize. TextGrad

mirrors PyTorch's abstraction for backpropagation, enabling optimization

across black-box AI systems. **d**–**g**, Example applications of TextGrad to

optimize molecular structures for properties such as drug-likeness and protein binding affinity (**d**), optimize solutions to coding problems (**e**), optimize

radiotherapy treatment plans (**f**) and optimize prompts to improve the reasoning of language models (**g**).

**610**  |  Nature  |  Vol 639  | 20 March 2025

**a**

▽LLM(Prediction, Evaluation)

▽LLM(Prompt, Prediction,![][image22])

*∂*Evaluation            “To achieve the desired change  *∂*Evaluation             “Based on the evaluation,	*∂*Prompt     ·        in the prediction, you should       *∂*Prediction   ·        the prediction is incorrect.

Prompt![][image23]Prediction                                                           Evaluation

![][image24]  LLM

use a different strategy... ”                                             To ﬁx it, you should change... ”

	*∂*Prediction  

*∂*Prompt

*∂*Evaluation

*∂*Evaluation

*∂*Evaluation

*∂*Prediction

*∂*Prediction

![][image25]  LLM

*∂*Prompt

∆

\=

∆

\=

*\=*

**b**

![][image26]  \=  ▽LLM (*x*, *y,* ![][image27])  ![][image28] “Here  is  a  conversation  with  an  LLM:  {*x*|*y*}. ”

\+

LLM (Here  is  a  conversation  with  an  LLM:  {*x*|*y*}.

Below  are  the  criticisms  on  {*y*}:

*∂L*

*∂y*

Explain  how  to  improve  {*x*}.)

**c**

![][image29]  LLM (Below  are  the  criticisms  on  {*x*}:

*∂L*

*∂x*

Incorporate the criticisms and produce a new variable.)

*x*new \= TGD.step(*x*, ![][image30])

**Fig. 2 | Illustrative implementation ofTGDand TextGrad’s backpropagation.**

**a**, We illustrate how backpropagation works fora system of two language model calls. It is noted that we overload the notation to denote both cases when the  
output variable does and does not have successors. Here, green text indicates the parameter we want to optimize. **b**, We show an example implementation of the textual gradient operator. This operator provides a natural language

feedback critique to the variable we want to optimize, where the critique

describeshow to modify the variable to improve the overall system. Here, the

purple text indicates where an LLMisusedin these operations. **c**, Example

implementation of one TGD iteration. Given the current variable and the textual gradients (feedback) we collected for this variable, the optimizer seeks to update this variable to improve the whole system. The exact prompts we use are different to ensure generality and flexibility; we use these examples only for exposition.

**Backpropagating LLM feedback with TextGrad**

We first describe an example to demonstrate what TextGrad looks like for a system of two LLM calls, and give the more general form in Methods. Consider the following system:

Prediction \= LLM(Prompt+ Question),                       (1)

Evaluation \= LLM(Evaluation Instruction+ Prediction),        (2)

where we use \+ to denote the concatenation of two strings and use LLM(*x*) to give*x* asa prompt toa language model to collect the response. This structure is used frequently where a language model is used for evaluation, for example, in LLM-as-a-judge[16](#bookmark=id.64o75th0a9mz).

In traditional automatic differentiation, we compute gradients using the chain rule to determine how to modify the numerical variables ina system to improve the system with respect to an objective function. In comparison, TextGrad generates feedback in the form of text to guide the modifications of unstructured variables in a system. It does not compute numerical gradients, but rather provides critiques suggesting specific modifications that would improve the system with respect to an objective function.

To improve anAI system, we instantiate an analogue of the backpro- pagation[14](#bookmark=id.64o75th0a9mz)algorithm (Fig. [2a](#bookmark=id.64o75th0a9mz)). Here, we use ∇LLM forthe textual gradient

operator when the forward function is an LLM call. A flexible way to instantiate ∇LLM can be found in Fig. [2b](#bookmark=id.64o75th0a9mz). In particular, this function returns a critique such as “The prediction can be improved by…” where the feedback describeshow to modify a variable to improve the overall system with respect to an objective function, analogously to numerical gradients. In this context, a modifiable component can bea prompt, an intermediate output orthe final prediction. In Fig. [2a](#bookmark=id.64o75th0a9mz), to backpropagate feedback to the prompt, we first collect feedback to the prediction variable using the evaluation. Then, given this feedback and the (LLM(Prompt \+ Question)) call, we collect the feedback on the prompt. More generally, we apply this procedure using the feedback obtained for all successors ofa variable.  
In numerical gradient descent, the parameter *θ* is combined with

![][image31],l![][image32],![][image33]e, ![][image34]![][image35]![][image36]g.![][image37]![][image38]i![][image39]![][image40]![][image41]![][image42]i![][image43],![][image44]\-![][image45]![][image46]i

the gradients with respect toa loss function L by subtraction, for exam-

updated using the feedback. In Fig. [2c](#bookmark=id.64o75th0a9mz), we showa concrete way to instan-

tiate TGD. This step uses the textual gradient to modify components

in ways that improve the system with respect to the objective function.

In automatic differentiation, the objective function is typically a

differentiable function, such as mean-squared error. In TextGrad,

the objective can be a complex and potentially non-differentiable or

black-box function, where the domain and codomain of the function

can be unstructured data. This choice adds important generality and

**Article**

flexibility to the framework. For instance, we show thatthe objective function can be obtained by prompting a language model, an output of a code interpreter running unit tests or outputs of molecular simu- lation engines. A simple objective function using an LLM can look like the following:

Loss (question, solution)=

LLM (“Below is a multi-choice question and a solution: {question, solution}

Yourjob isto investigate the solution.

Critically go through reasoning steps, and.

see if the prediction could be incorrect.”),

where we can use this evaluation signal to optimize the code snippet, powered by the well-documented potential of LLMs to self-critique and self-improve[17](#bookmark=id.64o75th0a9mz)–[22](#bookmark=id.64o75th0a9mz) .

We use TextGrad to solve two broad categories of tasks. First, in test-time optimization, we directly treat a solution to a problem—for example, a code snippet, the solution toa question ora molecule—asan optimization variable. For instance, in the above question-answering objective function, we have a solution that we would like to improve attest time. Our framework produces the textual gradients for and iteratively improves the solution variable. Second, in prompt opti- mization, the goal istofind a prompt that improves the performance of an LLMin multiple queries for a task. For example, we may want to find a system prompt to an LLM that improves the performance on mathematical reasoning questions. In particular, we want the system prompt to generalize to work for new problems, in contrast to test-time optimization where the only goal is to improve a variable fortheprob- lem at hand attest time. Crucially, both types of problem are solved without handcrafting the framework.

We demonstrate the flexibility of TextGradina diverse array of appli- cations. First, we optimize code snippets to solve hard problems from LeetCode, an online programming learning platform. Next, we optimize solutions to scientific questions. Third, we optimize prompts to guide the reasoning of LLMs. Fourth, we optimize radiotherapy treatment plans for patients with prostate cancer. Finally, we optimize compound systems involving multiple variablesand longer chains. Furthermore, in Supplementary Note 8.1, we discuss how we optimize chemicalstruc- tures for molecular property maximization as a case study. Across a dataset of 58 target proteins, TextGrad shows improved performance in ligand design compared with alternative molecule optimization methods (Extended Data Figs. 1 and 2).

**Code optimization**

In code optimization, the goal is to modify a code to improve its cor- rectness or runtime complexity. We often have a computation graph like the following:

Refinement objective

\= LLM (Problem \+Code+ Test-time instruction \+Test results),

where we optimize the code to solve a given problem with limited, local unit-test supervision, and self-evaluation through a test instruction asking to critique the current iteration of the code. Figure [1e](#bookmark=id.64o75th0a9mz)shows an example problem: “You are given an array numsofsize*n* consisting of distinct integers from 1 to*n*anda positive integer *k*. Return the number of non-empty subarrays in nums that have a median equal to*k*”. The first solution proposed by gpt-4o does not pass the tests. TextGrad identifies an edge case in the first solution and provides a suggestion on how to improve it. The optimized implementation passes all tests.	WeusetheLeetCode Hard dataset[23](#bookmark=id.64o75th0a9mz)to benchmark code optimization. LeetCode isan online platform that offers coding practice questions in

preparation for technical interviews. The dataset contains examples of hard coding problems that are challenging for both humans and language models, where the success metric is completion rate (pass- ing all test cases).

In LeetCode Hard, using gpt-4o ina zero-shot setting gets a 26% com- pletion rate, whereas Reflexion[23](#bookmark=id.64o75th0a9mz), a state-of-the-art self-refinement baseline, gets a 31% completion rate (Extended Data Fig. 3). With Text- Grad, we achieve a completion rate of 36%. These improvements are more significant considering that Reflexion was run with in-context demonstrations and TextGrad did not use any demonstrations (that is, zero shot). We provide more details around the dataset, baselines and promptsin Supplementary Note 3\.

**Solution optimization**

In solution optimization, the goal is to improve the solution toa prob- lem such as a question about quantum mechanics. We have the fol- lowing relation:

Solution refinement objective

\= LLM (Question+ Solution \+Test-time instruction),

where the parameter we optimize is the solution, and the loss func- tionis obtained by an evaluation of the solution. In each iteration, the LLM is prompted with the question, current solution and a test-time instruction asking to critique the current iteration. Although an LLM may not get the solution to a problem right atthe first attempt, it can improve the solution through iterative refinement[19](#bookmark=id.64o75th0a9mz),[23](#bookmark=id.64o75th0a9mz).

We explore solution optimization in question-answering datasets. We use Google-proof Question Answering (GPQA)[24](#bookmark=id.64o75th0a9mz), a recent benchmark where multiple-choice questions in physics, biology and chemistry are created and labelled by domain experts who have or are pursuing PhD degrees; two challenging subsets (machine learning and college physics) of the Massive Multitask Language Understanding (MMLU)[25](#bookmark=id.64o75th0a9mz) question-answering benchmark that is used to track the progress of language modelling with undergraduate-level questions; MathVista[26](#bookmark=id.64o75th0a9mz) and ScienceQA[27](#bookmark=id.64o75th0a9mz)to test the visual question-answering ability in scien- tific domains.

With TextGrad, we improve the performance of gpt-4oacross these benchmarks (Extended Data Fig. 4a). We improve the performance in MMLU subsets from 85.7% to 88.4% (machine learning) and from 91.2% to 95.1% (college physics), and GPQA from 51.0% to 55.0%. For multi- modal reasoning tasks, compared with Reflexion, TextGrad achieves performance gains of2.1% and 2.2% on MathVista and ScienceQA-IMG (the multimodal setting of ScienceQA). We provide further details in Supplementary Note 4\.

**Prompt optimization for reasoning**

LLMs show impressive performance in reasoning tasks, but their per- formance can be sensitive to the prompt used to guide their behaviour. With the right prompt, their reasoning can be significantly improved[28](#bookmark=id.64o75th0a9mz). In prompt optimization, the goal istofinda prompt to guide thebehav- iourofan LLM such that it performs well on a given task. Often, we have a computation graph like the following:

Answer \= LLM (Prompt, Question),

(3)

Evaluation \=Evaluator (Answer, Ground truth) ,

where we have a question, an answer to the question, and an evaluation indicating the quality of the output given the ground-truth answer. For instance, for a question-answering task, the evaluation metric could bethe accuracy of the answer.

Here, given a handful of training examples to optimize a prompt, the goal is to maximize the performance of an LLM onthe given task.

In our experiments, our goal is to improve the performance of a weaker and cheaper model (for example, gpt-3.5-turbo) using the feedback generated by stronger models (for example, gpt-4o). This task is useful in practice because after paying a fixed cost to optimize a prompt, the prompt-optimized weaker model can be used with cheaper inference costs instead of the stronger and more expensive model. Notably, we can decompose a prompt into instruction and in-context examples[1](#bookmark=id.64o75th0a9mz), and apply the same strategy to optimize them jointly using textual gradients. For instance, we can initialize the in-context examples using input–output samples from the training dataset, then optimize them using TGD.

We use two standard reasoning tasks (object counting and word sorting) from Big Bench Hard[29](#bookmark=id.64o75th0a9mz) and the GSM8k grade-school maths problem-solving dataset[30](#bookmark=id.64o75th0a9mz) . Across the three datasets, TextGrad improves the performance of the zero-shot prompt (Extended Data Fig. 5a). TextGrad outperforms OPRO[31](#bookmark=id.64o75th0a9mz), performs similarly to the state-of-the-art prompt optimization framework DSPy[5](#bookmark=id.64o75th0a9mz)  for word sorting and GSM8k, and improves over it by 7.0% for object counting. We report ablations in Supplementary Table 1, finding that (1)jointly optimizing demonstrations and the instruction ina prompt improves the performance the most, and (2) optimizing the instruction alone still outperforms other existing methods. We provide examples of optimized instructions and demonstrations in Extended Data Fig. 5\.

We perform further ablationsto analyse the scalability and robust- ness of TextGrad. In Supplementary Note 6.1, we perform optimization using Llama-3.1-70B[32](#bookmark=id.64o75th0a9mz), finding that optimizing the prompts using Text- Grad for less capable open-source models is also effective. In Supple- mentary Note 6.2, we study the robustness of the optimized prompts, finding that prompts optimized for gpt-3.5-turbo can be transferred to open-source models such as Qwen-2.5[33](#bookmark=id.64o75th0a9mz)and Llama-3.2-9B[32](#bookmark=id.64o75th0a9mz)to improve their performance. Finally, in Supplementary Note 6.3, we study the cost–accuracy frontier of the methods, finding that TextGrad brings Pareto improvements.

**Radiotherapy treatment plan optimization**

Radiotherapy is a cancer treatment that uses beams of intense energy, such as X-rays, to kill cancer cells. Before treatment, a team of radiation oncologistsand planners collaborate to design an effective treatment plan. This involves determining the necessary dose of radiotherapy and pinpointing the exact locations that need treatment.

Treatment planning involves essentially a two-loop multi-objective optimization problem, where the inner loop aims to optimize theradia- tion dose delivery to the tumour (planning target volume (PTV)) while protecting healthy organs (organs at risk (OARs)), given the impor- tance weights. The outer loop optimizes these importance weights to trade-off treatment effectiveness with protection. The main challenge lies in translating clinical goals into the weighted objective function. Currently, planners rely on time-consuming trial-and-error approaches to adjust importance weights, making the process time-consuming and costly[34](#bookmark=id.64o75th0a9mz),[35](#bookmark=id.64o75th0a9mz).

We apply TextGrad to perform the outer loop optimization. We per- form this optimization attest time with gpt-4o over the importance weights represented asa string, for example, *θ* \= “weight for PTV: \[PTV WEIGHT\], weight for bladder: \[BLADDER WEIGHT\], weight for rectum: \[RECTUM WEIGHT\], weight for femoral heads: \[FH WEIGHT\], weight for body: \[BODY WEIGHT\]”. The evaluation uses the treatment plan *P* and the clinical goals *g* using an LLM:

L(*θ*) \= LLM(*P*(*θ*), *g*),

and the new importance weights are obtained by performing TGD. To further improve the LLM’s capability to understand the relationship between the importance weights *θ* and the resulting plan *P* from the inner-loop optimizer, we provide a set of paired plans and their

corresponding importance weights {(*Pi*, *θi*![][image47] forTGD. To evaluate a treatment plan, we adopt several commonly used metrics. We consider the mean dose delivered to the target/organ volume, as well as*Dq*, which denotes the minimum dose received by *q*% of the target/organ volume.

First, weshowthatthe textual gradients generated by TextGrad pro- vide meaningful feedback for this problem. As shown in Fig. [3a](#bookmark=id.64o75th0a9mz), when thereis dosespillage outside thePTV, the textual gradient suggests an increase in the importance weight forthePTV. This adjustment results ina more uniform and confined dose forthePTV. However, this can lead to insufficient protection of the bladder and rectum as their relative weights are reduced. Therefore, in the following step, the feedback suggests slightly increasing the weights for the bladder and rectum, resulting in better protection (Fig. [3b](#bookmark=id.64o75th0a9mz),[c](#bookmark=id.64o75th0a9mz)).

We compared TextGrad-optimized plans with clinically approved plans used to treat five patients with prostate cancer. Although auto- mated planning systems exist[36](#bookmark=id.64o75th0a9mz) –[39](#bookmark=id.64o75th0a9mz), they have yet to become standard practice and cannot be used asa reliable source of ground truth. There- fore, manually designed and clinically approved plans continue to serve asthe gold standard. In Fig. [3d](#bookmark=id.64o75th0a9mz), we assess TextGrad’s capabilities in achieving clinical goals for the PTV region. TextGrad outperforms clinical plans, achieving a higher mean dose, and a*D*95 that exactly matches the prescribed dose, where *D*95 denotes the minimum dose received by 95% of the target/organ volume. In Fig. [3e](#bookmark=id.64o75th0a9mz), we focus on the protection of healthy organs. TextGrad-optimized plans achieve lower mean doses for these healthy organs, suggesting better organ protec- tion than human-optimized plans. Overall, these results arepresented as an illustrative demonstration of TextGrad’s potential rather than a definitive statement on clinical effectiveness. In Extended Data Fig. 6, we show that TextGrad also outperformsa Bayesian optimization algo- rithm, and give further details in Methods.

**Optimizing compound AI systems**

Finally, we investigate the ability of TextGrad to optimize compound AI systems involving multiple variablesand longer chains of reasoning.		Our first study focuses on Chameleon[40](#bookmark=id.64o75th0a9mz), a sophisticated agentic system with multiple LLM modules using diverse tools. Chameleon integrates various tools to tackle a broad range of tasks, leveraging abilities such as visual perception and knowledge retrieval. However, its original planning and execution pipeline is static with no feedback mechanism, which can lead to error accumulation asthe steps progress.	To address this, we use TextGrad to optimize the solution steps provided by Chameleon. This process starts with a standard forward pass through all modules. Subsequently, it iteratively optimizes each module’s output, starting from the first module and progressing to the last. For each module, the optimization process evaluates the final response using a loss function, computes gradients via TextGrad and backpropagates these gradientsto update the module’s output. By introducing this feedback loop, we address the limitations of static execution in the original Chameleon system. We evaluate this approach using ScienceQA-IMG, which requires the coordination of multiple tools in visual perception and knowledge retrieval (Supplementary Note 9).

The results in Fig. [4](#bookmark=id.64o75th0a9mz)show the effectiveness of TextGradin optimizing Chameleon. Through the identification of errors in reasoning traces and iterative backpropagation of textual gradientsto refine module outputs (as shown in Fig. [4a,b](#bookmark=id.64o75th0a9mz)), TextGrad achievesa 5.7% increase over the baseline performance with 2 optimization iterations, and a 7.7% gain with 3 iterations (Fig. [4e](#bookmark=id.64o75th0a9mz)). These findings highlight the potential of TextGrad to enhance compound systems like Chameleon.

In our second study, we use TextGrad to improve the spatial reasoning ability of multimodal models. In visual question answering, images are presented alongside questions related to the content of the images. To guide the model in answering the questions, we provide instructional prompts, such as “Answer the question based on the image, let’s think

**Article**

| Gradients: The current weight for the rectum and bladder are relatively low, which is not sufﬁcient to protect the rectum and bladder from receiving higher doses. Recommendation: Slightly   increase the importance weight for the bladder and rectum. |
| :---- |

![][image48]

Initialization                                   Iteration 1                                                        Iteration 5

Bladder

![][image49]

Rectum

Low ![][image50] High

Dose

**c**

**b**

40

90

|   ![][image51] |
| :---- |

85

35

80

30

75

25

70

20

65

15

60

10

55

5

50

0

5

4

3

2

1

Initial

**e**

**d**

Mean dose (Gy)

D95 (Gy)

Method

Target

70.20

\+1.97 (0.36)

**\+0.51** (0.09)

70.20

–0.10 (0.15) **\+0.00** (0.00)

Clinical goal

Radiation oncologist

TextGrad

PTV

| ![][image52] ![][image53] |  |
| :---- | :---- |
| ![][image54] |  |
| ![][image55] |  |

Initial         1            2            3            4            5

Iterations

**a**

| Gradients: The current plan does not meet the criteria for the PTV. The constraints on OARs are currently being met. Recommendation: Increase the importance weight for the PTV. |
| :---- |

Iterations

| Organ | Method | Mean dose (Gy) , |
| :---- | :---- | :---- |
| Bladder | Radiation oncologist TextGrad | 22.39 (5.55) **20.92** (0.79) |
| Rectum | Radiation oncologist TextGrad | 23.88 (6.45) **17.18** (4.20) |

**Fig. 3 | Radiotherapy treatment plan optimization.** We visualize the

backpropagated textual gradients adjusting importance weights of PTVsand OARs to balance tumour targeting and protection of OARs. **a**, The evolution

of a plan from initialization to iteration 5\. **b**, TextGrad iteratively improves the mean dose and reduces the dose variance of the PTV, achieving the clinical goal. **c**, TextGrad keeps the exposure for the bladder and rectum below clinically

allowed maximums. **d**,**e**, TextGrad-optimized plans have better dose metrics

(mean dose and *D*95) for PTV than clinically optimized plans (**d**), and lower doses on the bladder and rectum, indicating better protection of OARs (**e**). Overall,

these results arepresented asan illustration of TextGrad’s potential rather than a statement on actual clinical effectiveness. Values in bold indicate the best  
performing scores.

step by step”. Another language model evaluates the solution using a loss function prompt. Here we use TextGrad to jointly optimize the two sets of prompts—instructional prompt for the answer generation and the loss prompt for the evaluation—as well asthe generated solu- tion itself. All of these can be treated as separate variables in TextGrad, and optimized jointly. For evaluation, we use High-Quality Hallucina- tion Benchmark (HQH)[41](#bookmark=id.64o75th0a9mz), a spatial reasoning dataset with 1,600 visual question-answering pairs. The results in Fig. [4f](#bookmark=id.64o75th0a9mz)show that optimizing solutionsand prompts jointly with TextGrad achieves a 9% improve- ment over gpt-4o with chain of thought[42](#bookmark=id.64o75th0a9mz)and outperforms Reflexion.

**Discussion**

TextGrad is built on three principles: (1) it is a general and perfor- mant framework that is not handcrafted for a specific application domain; (2) it is easy to use, mirroring PyTorch abstractions, thus allowing knowledge transfer; and (3) it is fully open source. Through TextGrad, we obtained state-of-the-art results in code optimization and PhD-level question answering, optimized prompts, and provided proof-of-concept results in optimizing molecules and treatment plans.

**a**

Planner    ![][image56]

Image captioner

Solution generator

Answer generator

Query

![][image57]

Knowledge retrieval

Text  
detector

Input objects:

**b**

Tortoise shell Crown

Basketball

ariable

We need to evaluate each of the given options. \[...\] Next, we can summarize the ﬁndings: All three objects can be  
described as shiny. None of the objects are consistently  
slippery. All three objects are opaque. \[...\] We conclude that the common property is shiny.The answer is A.

Question: Which  
property do these  
three objects have in common?

Choices:

(A) Shiny

(B) Slippery (C) Opaque

Ground truth:  
The answer is (C) opaque.

ariable

This question involves comparing the properties of three

objects \[...\] Tortoise shells are typically hard and can have a shiny appearance, crowns are usually shiny, and basketballs are usually made of rubber or synthetic materials that can  
also have a shiny ﬁnish. \[...\]

Gradients

The solution is not accurate. Let's analyze the image step by step. 1\. Tortoise Shell: Typically, a tortoise shell is not

shiny; it is more matte and textured. 2\. Crown: Crowns are often made of metal and can be shiny due to their relective surfaces. 3\. Basketball: A basketball is not shiny; it has a  
textured surface for grip. \[...\]

**c**

Gradients

The knowledge could be improved by briely outlining the

key properties of the objects being compared. For instance, mentioning that tortoise shells are hard and opaque,  
crowns can be shiny and decorative, and basketballs  
are designed for grip and visibility would provide a more informative foundation for the subsequent analysis. \[...\]

ariable

| ![][image58] Instruction |
| :---- |

Iterative update

Image                   Question

ariable

| ![][image59] Answer |
| :---- |

ariable

| ![][image60] Loss instruction |
| :---- |

![][image61]

![][image62]

Loss

**d**

**e**

Prompt optimization

ariable

You will answer a question based on the image(s). Think step by step.

Prompt at initialization (accuracy \= 70%)

| Initial answer |
| :---- |
| A stop sign. |

ariable

Analyze the image carefully and answer the question by following these steps:

1\. Image Quality and Lighting:\[...\] 2\. Perspective Deﬁnition:  \[...\] 3\. Detailed Observation: \[...\] 4\. Spatial Awareness: \[...\]. 5\. Contextual Understanding: \[...\] 6\. Veriﬁcation and  
Cross-Veriﬁcation: \[...\] 7.Acknowledge Uncertainty and

Alternatives: Identify potential ambiguities and suggest

possible interpretations for unclear objects. 8\. Iterative Review:

Revisit observations after considering contextual factors.

Encourage multiple rounds of observation and cross-veriﬁcation.\[...\]

Prompt after optimization (accuracy \= 79%)

**f**

| Dataset | Method | Accuracy (%) |
| :---- | :---- | :---- |
| HQH | Chain of thought Relexion TextGrad (prompt and solution optimization) | 70.0 72.0 **79.0** |

Solution optimization

Question:

What is to the top right of

the bus?

| Answer after 3 iterations |
| :---- |
| To the right of the front of the bus, there is a tall, red pole. |

![][image63]

Gradients

The object to the right of the front of the bus is not a stop sign. It appears to be a bus stop pole or  
sign, which is different from a stop sign. The object type is incorrectly identiﬁed.

| Dataset | Method | Accuracy (%) |
| :---- | :---- | :---- |
| ScienceQA-IMG | Chameleon TextGrad (2 iterations of optimization) TextGrad (3 iterations of optimization) | 77.5 83.2 **85.2** |

**Fig. 4 | Optimizing compound AI systems. a**, TextGrad enhances the solution

generated by Chameleon, an agentic system that generatesa plan followed

by tool execution. We treat the output of each module asa separate variable

to optimize and improve them. **b**, Textual gradients are computed and

backpropagated through the tool execution chain. **c**, TextGrad iteratively

optimizes the answer toa multimodal question. To improve the spatial

reasoning ability of multimodal language models in visual question answering,

we treat loss prompt, the instruction and the solution as variables to jointly

optimize and improve them. **d**, With an example in HQH, we show how TextGrad jointly optimizes the promptsand solutions. **e**,**f**, TextGrad yields accuracy

improvements of 7.7% and 9% on ScienceQA-IMG (**e**) andon HQH (**f**), respectively. Values in bold indicate the best performing scores. Bus image in**d** reproduced  from ref. [41](#bookmark=id.64o75th0a9mz)under a Creative Commons licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Article**

Previous studies have explored using LLMsas optimizers for prompt engineering and self-improvement[5](#bookmark=id.64o75th0a9mz),[23](#bookmark=id.64o75th0a9mz),[31](#bookmark=id.64o75th0a9mz). Although these frameworks demonstrated the capabilities of LLMs in specific domains such as prompt optimization, our work introducesa general framework based uniquelyon backpropagating natural language feedback that extends beyond prompts to optimize diverse components including code, molecules and treatment plans.

There are various limitations that motivate future work to realize the potential of optimization frameworks powered by LLMs. We hope that TextGrad can be used to accelerate iterative processes in scientific discovery and increase the productivity of engineering efforts. Second, there is a large design space for algorithms to backpropagate textual feedback. We believe that there are many connections to be drawn between numerical optimization, automatic differentiation and Text- Grad. Finally, although we conducted proof-of-concept applications of TextGrad to design molecules and treatment plans with in silico validations, the ultimate test requires experimental and clinical assess- ments, which are outside of the scope of this paper.

AsAI systems shift from training individual models to optimizing compound systems with multiple interacting LLM components and tools, we need a new generation of optimizers. TextGrad combines the reasoning power of LLMs with the decomposable efficiency of backpropagation to create a general framework to optimize AI systems.

**Online content**

Any methods, additional references, Nature Portfolio reporting summa- ries, source data, extended data, supplementary information, acknowl- edgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at [https://doi.org/10.1038/s41586-025-08661-4](https://doi.org/10.1038/s41586-025-08661-4).

1\.      Brown, T. et al. Language models are few-shot learners. *Adv. Neural Inf. Process. Syst.* **33**, 1877–1901 (2020).

2\.     Trinh, T. H., Wu, Y., Le, Q. V., He, H. & Luong, T. Solving olympiad geometry without human demonstrations. *Nature* **625**, 476–482 (2024).

3\.     Li, Y. et al. Competition-level code generation with alphacode. *Science* **378**, 1092–1097 (2022).

4\.     Yang, J. et al. SWE-agent: agent–computer interfaces enable automated software engineering. In*Adv. Neural Inf. Process. Syst. 37* (NeurIPS, 2024).

5\.     Khattab, O. et al. DSPy: Compiling declarative language model calls into state-of-the-art pipelines. In *The Twelfth International Conference on Learning Representations* (2024).

6\.     Zaharia, M. et al. The shift from models to compound AI systems. *BAIR* [https://bair.berkeley.](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) [edu/blog/2024/02/18/compound-ai-systems/](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) (2024).

7\.      Zhou, Y. et al. Large language models are human-level prompt engineers. In *The Eleventh International Conference on Learning Representations* (2023).

8\.     Krizhevsky, A., Sutskever, I. & Hinton, G. E. Imagenet classification with deep convolutional neural networks. In*Adv. Neural Inf. Process. Syst.25* (NeurIPS, 2012).

9\.     Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* **596**, 583–589 (2021).

10\.    Fawzi, A. et al. Discovering faster matrix multiplication algorithms with reinforcement learning. *Nature* **610**, 47–53 (2022).

11\.     Mankowitz, D. J. et al. Faster sorting algorithms discovered using deep reinforcement learning. *Nature* **618**, 257–263 (2023).

12\.    Merchant, A. et al. Scaling deep learning for materials discovery. *Nature* **624**, 80–85 (2023).

13\.    Goodfellow, I., Bengio, Y. & Courville, A. *Deep Learning* (MIT Press, 2016).

14\.    Rumelhart, D. E., Hinton, G. E. & Williams, R. J. Learning representations by back- propagating errors. *Nature* **323**, 533–536 (1986).

15\.    Pryzant, R. et al. Automatic prompt optimization with “gradient descent” and beam

search. In *Proc. 2023 Conference on Empirical Methods in Natural Language Processing* (eds Bouamor, H. et al.) 7957–7968 (Association for Computational Linguistics, 2023).

16\.    Zheng, L. et al. Judging LLM-as-a-judge with MH-bench and chatbot arena. *Adv. Neural Inf. Process. Syst.* **36**, 46595–46623 (2023).

17\.    Li, X. et al. AlpacaEval: an automatic evaluator of instruction-following models. *GitHub* [https://github.com/tatsu-lab/alpaca\_eval](https://github.com/tatsu-lab/alpaca_eval) (2023).

18\.    Bai, Y. et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. Preprint at [https://arxiv.org/abs/2204.05862](https://arxiv.org/abs/2204.05862) (2022).

19\.    Madaan, A. et al. Self-refine: iterative refinement with self-feedback. In*Adv. Neural Inf.*

*Process. Syst. 36* (NeurIPS, 2023).

20\.   Stiennon, N. et al. Learning to summarize with human feedback. In*Adv. Neural Inf. Process. Syst.* **33**, 3008–3021 (2020).

21\.    Yuan, W. et al. Self-rewarding language models. In *Forty-first International Conference on Machine Learning* (2024).

22\.   Dubois, Y. et al. AlpacaFarm: a simulation framework for methods that learn from human feedback. In*Adv. Neural Inf. Process. Syst. 36* (NeurIPS, 2023).

23\.   Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K. & Yao, S. Reflexion: language agents with verbal reinforcement learning. *Adv. Neural Inf. Process. Syst.* **36**, 8634–8652 (2023).

24\.   Rein, D. et al. GPQA: a graduate-level Google-proof Q\&A benchmark. In *First Conference on Language Modeling* (2024).

25\.   Hendrycks, D. et al. Measuring massive multitask language understanding. In *The Ninth International Conference on Learning Representations* (2021).

26\.   Lu, P. et al. MathVista: evaluating mathematical reasoning of foundation models in visual contexts. In *The Twelfth International Conference on Learning Representations* (2024).

27\.    Lu, P. et al. Learn to explain: multimodal reasoning via thought chains for science question answering. *Adv. Neural Inf. Process. Syst.* **35**, 2507–2521 (2022).

28\.   Liu, P. et al. Pre-train, prompt, and predict: a systematic survey of prompting methods in natural language processing. *ACM Comput. Surv.* **55**, 1–35 (2023).

29\.   Suzgun, M. et al. Challenging BIG-bench tasks and whether chain-of-thought can solve

them. In *Findings ofthe Association for Computational Linguistics: ACL 2023* 13003–13051 (Association for Computational Linguistics, 2023).

30\.   Cobbe, K. et al. Training verifiers to solve math word problems. Preprint at [https://arxiv.](https://arxiv.org/abs/2110.14168) [org/abs/2110.14168](https://arxiv.org/abs/2110.14168) (2021).

31\.    Yang, C. et al. Large language models as optimizers. In *The Twelfth International Conference on Learning Representations* (2024).

32\.   Dubey, A. et al. The Llama 3 herd of models. Preprint at [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) (2024).

33\.   Yang, A. et al. Qwen2 technical report. Preprint at [https://arxiv.org/abs/2407.10671](https://arxiv.org/abs/2407.10671) (2024).

34\.   Khan, F. M., Gibbons, J. P. & Sperduto, P. W. *Khan’s Treatment Planning in Radiation Oncology* (Lippincott Williams & Wilkins (Wolters Kluwer), 2016).

35\.   Hussein, M., Heijmen, B. J. M., Verellen, D. & Nisbet, A. Automation in intensity modulated radiotherapy treatment planning—a review of recent innovations. *Br. J. Radiol.* **91**,

20180270 (2018).

36\.   Kisling, K. et al. Radiation planning assistant-a streamlined, fully automated radiotherapy treatment planning system. *J. Vis. Exp.* **134**, e57411 (2018).

37\.    Huang, C., Nomura, Y., Yang, Y. & Xing, L. Meta-optimization for fully automated radiation therapy treatment planning. *Phys. Med. Biol.* **67**, 055011 (2022).

38\.   Yang, Y. & Xing, L. Clinical knowledge-based inverse treatment planning. *Phys. Med. Biol.*

**49**, 5101 (2004).

39\.   Liu, S. et al. Automated radiotherapy treatment planning guided by gpt-4vision. Preprint at [https://arxiv.org/abs/2406.15609](https://arxiv.org/abs/2406.15609) (2024).

40\.   Lu, P. et al. Chameleon: plug-and-play compositional reasoning with large language models. *Adv. Neural Inf. Process. Syst.* **36**, 43447–43478 (2023).

41\.    Yan, B., Zhang, J., Yuan, Z., Shan, S. & Chen, X. Evaluating the quality of hallucination

benchmarks for large vision-language models. Preprint at [https://arxiv.org/abs/2406.17115](https://arxiv.org/abs/2406.17115) (2024).

42\.   Wei, J. et al. Chain-of-thought prompting elicits reasoning in large language models. *Adv. Neural Inf. Process. Syst.* **35**, 24824–24837 (2022).

**Publisher’s note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author

self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

© The Author(s), under exclusive licence to Springer Nature Limited 2025

**Methods**

Throughout the paper, unless otherwise specified, we use the gpt-4o-2024-05-13 version of gpt-4o and the gpt-3.5-turbo-0125 ver- sion of gpt-3.5.

**The general case for backpropagation**

The abstraction readily applies to arbitrarily complex systems. Define a computation graph by

*v* \=*fv* (PredecessorsOf (*v*))  ∀ *v* ∈ V,                        (4)

where *v*isa variable in the graph, Vis thesetofall variables in the graph, and SuccessorsOf returns the successorsand PredecessorsOf returns the predecessors ofa variable. Generally speaking, the value of *v* can be unstructured data, such as natural language text or images. Formost of the results and exposition in this paper, *v* is natural language text.

Furthermore, let us have*fv* as the transformation that consumes a set of variables and produces the variable *v*. For instance, we can use an LLM or numerical simulator as a transformation. As different functions will have different ways to compute gradients and collect feedback for, we will generally use ∇*f* to denote the gradient function for a function*f*. For the sake of exposition, we will omit the subscript when the function is obvious.

The gradients are computed by

![][image64]

where we collect thesetof gradients from all successor variables *w* of *v*. Intuitively, we get feedback from every context in which a variable *v* was used, and aggregate them.

The equation above recursively computes the gradientsofthe down- stream objective with respect to the desired variables *v* in the graph. The ∇*f* function takes as input the gradients ofL with respect to the successorsofa given variable *v*, the value of the variable *v*and thesuc- cessors themselves. It is noted that the final gradient variable comprises a set of contextsand feedback for anyplace a variable was used in.

Finally, to update any desired variable *v* in the graph, we can use an optimizer step:

![][image65]

which updates the value of *v* based on its current value and the gradi- ents. Fora computation graph where there are*n* edges, each iteration of optimization performs at most *n* additional language model calls to compute gradients (one call using the gradient operator for each edge in the computation graph). For the concrete implementations of operations in TextGrad, see Supplementary Note 1\. For the complete algorithm, see Algorithm 1 in Supplementary Note 1\.

**Code optimization**

Reflexion is the state-of-the-art method on the LeetCode Hard data- set. Their approach prompts an LLM to self-reflect on code snippets and the errors that were generated attest time using candidate unit tests. Given the self-reflection, the LLM is prompted again to provide an updated piece of code, conditioned on the self-reflection and the errors. We ran Reflexion on LeetCode Hard using gpt-4o using one in-context demonstration to guide the behaviour (one shot). In addition to Reflexion, we also runa zero-shot baseline using gpt-4o mimicking the same zero-shot baseline described in the Reflexion paper. In con- trast to Reflexion, TextGrad runs in a zero-shot setting, without any demonstrations. In Extended Data Fig. 3a, we show the code refinement objective used in this paper.

We ran the experiment five times with five different seeds and we averaged the results. At each iteration of optimization, TextGrad makes one call togpt-4oto evaluate the test-time loss, one call to collect tex- tual gradientsandone call to update the code snippet. The number of coding problems in LeetCode Hard is 39\.

**Solution optimization**

The objective function for the refinement can be found in Extended Data Fig. 4b. Similarly, we report the objective used for multimodal solution optimization in Supplementary Note 4\.

GPQA is a recent benchmark where challenging multiple-choice questions in physics, biology and chemistry are created and labelled by domain experts who have or are pursuing PhD degrees. In this bench- mark, experts and skilled non-experts are reported to achieve 81% and 22% accuracy, respectively, demonstrating the difficulty of the questions. Wealsouse two challenging subsets (machine learning and college physics) of the MMLU question-answering benchmark that is used to track the progress of language modelling and whether LLMs reached human-level performance. Here the expert human accuracy on average is around 90%.

We further test solution optimization in multimodal problems, spe- cifically the MathVista and ScienceQA tasks. MathVista is a comprehen- sive benchmark for evaluating models’capabilities in mathematical reasoning with visual contexts. It contains seven mathematical reason- ing categories including arithmetic reasoning, algebraic reasoning and statistical reasoning, and requires fine-grained visual understand- ing across natural images, diagrams, charts and graphs. We utilize the testmini split with 1,000 examples for evaluation. ScienceQA is a multimodal benchmark of multiple-choice questions in scientific domains. It covers three subjects: nature science, social science and language science, featuring a diverse set of domains and topics. We conduct experiments on ScienceQA-IMG, the test split in multimodal setting, which consists of 2,017 examples. We provide further details on format and promptsin Supplementary Note 4\.

We report three baselines. First, the reported results in the gpt-4o release document states 53.6% accuracy. However, their official imple- mentation uses a temperature of 0.5 for generations, thus we also test gpt-4owith temperature 0 and chain of thought (CoT) prompting provided in the official implementation. For TextGrad, we perform three iterationsoftest-time updates (that is, update the solution three times) and perform majority voting across all solutions to get the final answer. We use string-based metricsto compute the final accuracy of each answer. Finally, we report Reflexion as another baseline, where we also run their algorithm with three iterations (that is, update the answer three times) and perform majority voting across all solutions to get the final answer.

**Prompt optimization**

Here, we use TextGrad in a minibatch stochastic gradient descent setting[43](#bookmark=id.64o75th0a9mz). In particular, at each iteration, weusea few training examples for training. A pseudocode and short implementation can be found in the snippet above. Full details of prompt optimization can be found in Supplementary Note 5\.

Furthermore, we can decompose a prompt into an instruction and in-context demonstrations, and apply the same strategy to optimize them jointly using textual gradients. For instance, we can initialize the in-context demonstrations using input–output samples from the training dataset, then optimize them using TGD.

We explore improving the performance of gpt-3.5-turbo using gpt-4o to provide feedback during backpropagation. In particular, although the forward model that performs the reasoning is gpt-3.5-turbo, we use gpt-4oto provide the feedback and improve the prompt. We use a batch size of 3 with 12 iterations, that is, the model sees 36 training examples in total, sampled randomly with replacement. After each iteration, weruna validation loop with the validation set of the datasets,

**Article**

and if the performance is better than the previous iteration we update the prompt, which consistsofa separate instruction variable, and the in-context demonstrations.

We explore prompt optimization in multiple datasets, including the two standard reasoning tasks (object counting and word sorting) from Big Bench Hard (randomly split into 51/100/99 train/validation/ test samples) and GSM8k grade-school maths problem solving (using train/validation/test splits from DSPy). For GSM8k and object count- ing, we use a string-based exact match metric to quantify accuracy (that is, whether the final number provided in the response is the same as the ground-truth answer). For word sorting, we use an LLM to compare the response and the ground-truth answer. We give more details about the tasks, promptsand example queries in Supplemen- tary Note 5.1.

We have three main baselines:

1\. Zero-shot CoT. We initialize all prompts asa zero-shot CoT prompt, where amodelis instructed to ‘Think step by step’ to explain its rea- soning before giving its answer. This strategy is well known to be a strong baseline for prompting.

2\. DSPyisa state-of-the-art language model programming and prompt optimization framework; thus we use it as the reference baseline. We instantiate DSPy’s BootstrappedFewShotRandomSearch opti- mizer with ten candidate programs and eight few-shot examples. This optimizer identifies demonstrations to include in the prompt as few-shot examples. This selection is done through generating traces of LLM inputsand outputs that individually pass the metric (in this case, accuracy) and includesCoT reasoning. DSPy then app- lies random search over subsets of up to size eight shots with these demonstrations.  
3\. OPROis one of the first methods to use LLMs as optimizers, where given a set of promptsand their performance, an optimizerLLM gen- erates new candidate prompts. Weuseallofthe hyperparameters as reported in the paper. At each iteration OPRO generates 8 candidate prompts by prompting the optimizer LLM with 20 best-performing prompts along with their performance, and 3 exemplars from the dataset. We perform eight iterations ofthis procedure.

**Treatment planning**

The dataset used in this study comprised imaging and treat- ment plans for five patients with prostate cancer who underwent intensity-modulated radiation therapy. The available data for each patient includes computed tomography scans, delineated anatomical structures and clinically approved treatment plans obtained via Eclipse.

As we mentioned in the results, TextGrad is used to optimize the hyperparameters (for example, importance weights for PTVand OARs) of the inner-loop numerical optimizer that generates the treatment plan. This optimization is done using a variation of TextGrad, that is, ‘projected gradient descent with momentum updates’. In particular, three prostate cancer treatment plans optimized by clinicians, along with their corresponding hyperparameters, are provided as in-context examples. These examples guide the updatesofthe hyperparameters. Moreover, the historical hyperparameters and the textual gradients from past iterations, asan analogy to momentum, are also included in the prompts for updating the hyperparameters. This additional context helps refine the optimization process. The optimization will be stopped if the loss suggests all protocols meet, otherwise, it will be stopped if the maximum number of iterations (we set it toten) is reached.

The importance weights are all initialized at 100 for different organs. The dose objectives are set to 70.20 for PTV, 0.00 for bladder andrec- tum, and 30.00 for femoral heads and body, and fixed during optimi- zation.

To evaluate a treatment plan, we adopt several frequently used dose metrics, asa plan cannot be evaluated using a single metric. We consider the mean dose delivered to the target/organ volume, as well as*Dq*, which denotes the minimum dose received by *q*% of the target/organ volume.

Throughout our experiments, we compare the plans generated by TextGrad with multiple baselines. Our first baseline uses plans gener- ated by certified physicists from our institution with multiple years of experience in developing treatment plans across various cancer types. Our second baseline uses plans produced using Bayesian optimiza- tion, which optimizes hyperparameters *θ* (for example, importance weights for PTVand OARs) by employing a weighted sum of multiple plan evaluation metrics[44](#bookmark=id.64o75th0a9mz) as the objective function. Specifically, the objective is defined as

![][image66]

*F*(*m*(*θ*)*i* ; *mi*)= (*m*(*θ*)*i* − *mi*)2 and *αi* \=2−*pi* ,

where *mi* represents the plan evaluation metric derived from the gen- erated plan with hyperparameter *θ*, and *mi* denotes the corresponding clinical goal. The term*pi* indicates the rank of importance for each plan evaluation metric.  
	For PTV, we use the mean dose and*D*95 as evaluation metrics with  
*m* \=70.20, assigning them an importance rank *p* of 1\. For the OARs,

that is, bladder and rectum, we consider the mean dose, *D*5, and *D*50

with *m* \=0.00, and an importance rank *p* of 2\.

We implement Bayesian optimization using Optuna[45](#bookmark=id.64o75th0a9mz). The number of optimization trials in Optuna is set to 50\. For all hyperparameters, we configure the suggest\_float range to \[100, 400\], while leaving the remaining parameters as default.

In Extended Data Fig. 6, we show additional results on comparing TextGrad-optimized plan with clinicians optimized plans. Although Bayesian optimization can optimize the plan quality metric (PQM) score function, it produces significantly worse results, particularly in terms of dose conformity, and converges slowly, typically requiring an average of21.3 iterations. In contrast, our method, TextGrad, generates high-quality plans in just ten iterations. The results illustrate theeffec- tiveness of TextGradin guiding iterative improvements in optimization but do not constitute a direct evaluation of clinical utility.

**Optimizing compound AI systems**

**Optimizing the Chameleon agentic system.** The original Chameleon system is described in Algorithm 2 (Supplementary Note 9). It consists of a planner P and a module inventory of tools M. Given an input query *q*, the planner generates a plan *p* ← P(*q*; M) that specifies a sequence of modules to execute. Executing the plan starts from an empty cache *c* 0 \=∅. At each step *t*, the chosen module *Mt* is run as *yt* ← *Mt*(*q*; *ct*−1), after which the cache is updated: *ct* ← update\_cache(*ct*−1, *yt*). After all *T* steps, the final response is *r* \=*yT*.

We augment this procedure, as shown in Algorithm 3 (Supplementary Note 9), by introducing textual gradient-based optimization. After an initial forward pass, we iteratively refine each module’s output. We quantify the final response quality via a loss function L, compute tex- tual gradients using TextGrad and propagate these gradients back to earlier steps in the chain, adjusting their outputs accordingly. This iterative process gradually improves the intermediate reasoning steps and, consequently, the final answer. We evaluate the original and TextGrad-optimized systemson ScienceQA-IMG and experiment with two and three optimization iterations to study the impact of iteration count. Details are available at Supplementary Note 9\.

**Improving spatial reasoning ability by jointly optimizing prompts and solutions.** We apply TextGrad to improve spatial reasoning ability on the visual question-answering task. In visual question answering, instructional prompts are used to guide the model in answering the questions following. We further compute the objective in TextGrad by prompting a language model and treating both the prompts embed- dedin the loss function and the instructional prompts as variables to

be optimized. The generated solutions to the questions are optimized together.

We split the HQH dataset into 70/30/100 (train/validation/test) splits, out of the 200 samples in the relation question type. The complete optimization process is as follows. We first warm up the optimization process where the instructions to the language model that specify the task, and the instructions for the loss function used to evaluate gener- ated answers are optimized inan alternating fashion with a batch size set toten (in total seven iterations). We use the prompts that achieve the best validation accuracy for test-time optimization. Attest time, each solution is optimized for three iterations (that is, update the solu- tion three times). Majority voting across all solutions is performed to get the final answer.

**Molecule optimization**

As a proof of concept, we also explore TextGrad’s applications to molecular optimization, and demonstrate how our framework can be used to interface with domain-specific tools in scientific tasks (Sup- plementary Note 8).

We optimize chemical molecules encoded as text (simplified molecular input line entry system (SMILES) strings) with respect to one or several molecular scoring functions. We restrict our scoring functions to in silico methods, namely, Autodock Vina, which meas- ures the binding affinity of a molecule against a target protein, and quantitative estimate of drug-likeness (QED), which measures how ‘druglike’ a molecule is. We initialize a molecule as a small starting fragment, and perform several iterations of optimization using TGD. At each iteration, the loss is constructed by concatenating each score into a single ‘textual’ loss.

We evaluate our methods using two baselines: (1) molecular char- acteristics of approved drugs against a matched set of protein tar- gets (Extended Data Fig. 1); and (2) oracle efficiency compared with state-of-the-art molecular optimization algorithms (Extended Data Fig. 2). We observe that TextGrad produces molecules with a similar distribution to its clinical counterparts across both optimized and non-optimized molecular properties, but with low structural simi- larity to approved drugs or existing compounds. Compared with state-of-the-art molecular optimization algorithms, TextGrad con- sistently produces higher-performing molecules while using fewer evaluations on a diverse set of protein targets. Although promising, we caution that the practical value of these molecules is limited and should betakenasa proof of concept only. Our evaluations are limited

by their reliance onin silico property predictors, which are not accurate substitutes for thorough experimental and clinical evaluation.

**Data availability**

We used publicly available data to evaluate TextGrad. Details on how to access the data is available at [https://github.com/zou-group/textgrad](https://github.com/zou-group/textgrad).

**Code availability**

The TextGrad code and experiments are available at [https://github.](https://github.com/zou-group/textgrad) [com/zou-group/textgrad](https://github.com/zou-group/textgrad)and [https://doi.org/10.5281/zenodo.14497017](https://doi.org/10.5281/zenodo.14497017) (ref. [46](#bookmark=id.64o75th0a9mz)).

43\.   Bottou, L. Large-scale machine learning with stochastic gradient descent. In *Proc.*

*COMPSTAT’2010* (eds Lechevallier, Y. & Saporta, G.) 177–186 (Physica-Verlag, 2010).

44\.   Wang, Q. et al. High-dimensional automated radiation therapy treatment planning via bayesian optimization. *Med. Phys.* **50**, 3773–3787 (2023).

45\.   Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. Optuna: a next-generation

hyperparameter optimization framework. In *Proc. 25thACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (2019).

46\.   Bianchi, F. et al. zou-group/textgrad: v0.1.6. *Zenodo* [https://doi.org/10.5281/zenodo.14497017](https://doi.org/10.5281/zenodo.14497017) (2024).

**Acknowledgements** We thank D. Yilmaz, F. Dinc, B. Ergun, Y. Sun, I. Covert, K. Swanson,

O. Faruk Akgun, Y. Efe, O. Khattab, K. Y. Wu, E. Wu, K. Vodrahalli, O. Pastor Serrano, P. J. Chia,

J. Tagliabue, N. Thakkar, E. Simon, S. Eyuboglu, I. Gao, L. Chen and members of the Zou group and the Guestrin group for their support and comments on this work. This work was supported by funding from the Chan-Zuckerberg Biohub. C.G. was supported by funding from the

Chan-Zuckerberg Biohub, Stanford HAI, AFOSR Grant FA9550-21-1-0397, gifts from Google and IBM.

**Author contributions** M.Y. and J.Z. conceptualized the research and led the overall project.

M.Y. developed the primary codebase and led prompt optimization and solution optimization.

M.Y., F.B. and J.B. designed the abstractions. F.B. led code optimization. J.B. led molecule

optimization. S.L. led treatment planning optimization and compound system optimization.

P.L. led solution optimization in multimodal settings and compound system optimization.

Z.H. and C.G. advised the project. J.Z. supervised the project. All authors contributed to the preparation of the paper and approved the final version.

**Competing interests** The authors declare no competing interests.

**Additional information**

**Supplementary information** The online version contains supplementary material available at [https://doi.org/10.1038/s41586-025-08661-4](https://doi.org/10.1038/s41586-025-08661-4).

**Correspondence and requests for materials** should be addressed to Mert Yuksekgonul or James Zou.

**Peer review information** *Nature* thanks Kai-Wei Chang and the other, anonymous, reviewer(s) for their contribution to the peer review of this work. Peer reviewer reports are available.  
**Reprintsand permissions information** is available at [http://www.nature.com/reprints](http://www.nature.com/reprints).

**Article**

**a**

**b**

| ![][image67] ![][image68] Gradients: Introduce functional groups that can form hydrogen bonds or hydrophobic interactions. |
| :---- |

| ![][image69]Gradients: Add hydrophobic groups or aromatic rings to enhance interactions ... while maintaining a balance of hydrophobic and hydrophilic properties. |
| :---- |

![][image70]  
![][image71]

H  N 2 \\

\\

Vina: \-7.5 kcal/mol

Vina: \-5.5 kcal/mol

QED: 0.59

Vina: \-4.2 kcal/mol

QED: 0.44

QED: 0.79

**Molecule optimization**

|  \-2 \-3 \-4 \-5 \-6 \-7 \-8 \-9 \-10 \-11  二 亏 TextGrad   ![][image72] 0.0         0.2         0.4         0.6         0.8         1.0 druglikeness (QED) | c |  \-4 \-5 \-6 \-7 \-8 ![][image73] Gemfibrozil ★ ![][image74]Fenofibric acid Fenofibrate ![][image75] ![][image76] 0.4       0.5       0.6       0.7       0.8       0.9       1.0 druglikeness (QED) |
| ----- | :---- | ----- |

Fenofibric Acid (Fibricor)

**d**

![][image77]

![][image78] TextGrad  
![][image79]

Tanimoto: 0.38

Tversky: 0.36

**e**

PPARA (2p54)

**Extended Data Fig. 1 |** See next page for caption.

**Extended Data Fig. 1 | Molecule optimization via text.** TextGrad optimizes

a starting benzene fragment to improve its druglikeness (higher QED) and

binding affinity (lower vina score) to the protein receptor PPARA. The textual gradients for the first three iterations are shown in **(a)**, and the performance of all ten iterations compared to clinically approved molecules targeting

PPARA in **(c)**. The molecule atthe final iteration has low structural similarity

with its most similar clinically approved counterpart, and better QED and

Vina scores **(d)** with a highly plausible pose geometry shown in **(e)**. Across 29 targetsand three initial fragments, TextGrad successfully designs molecules with similar vina scoresand greater QED scores than clinically approved

molecules **(b)**.

**Article**

**b**

**a**

1.0

10

0.9

8

0.8

6

4

0.7

2

0.6

CDK2                                         CYP3A4                                         DHFR

| ![][image80] | 12 10 8 6 4 2 | ![][image81] | 12 10 8 6 4 2 | ![][image82] |
| :---- | :---- | :---- | :---- | :---- |

0                  20                 40                 0                  20                 40                 0                  20                 40 Oracle Evaluations

**Extended Data Fig. 2 | Comparing TextGrad toto other molecule**

**optimization methods. (a)** Scaled mean top-5 AUCs for each method across all 58 protein targets. **(b)** Sample trajectories. For each algorithm, we selected the best performing trajectory to visualize, as measured by scaled Top-5 AUC, for

the protein target listed, with the shaded error bars representing the standard error across the three repetitions (seeds/initial molecules). The blue star

indicates the iteration at which TextGrad’s early stopping condition was triggered.

**a**

Code Reﬁnement Objective

LLM ( “You are an intelligent assistant used as an evaluator, and part of an optimization system. You will analyze a code implementation for a coding problem and unit test results.  The code will be

tested with harder tests, so do not just check if the code passes the provided tests. Think about the correctness of the code and its performance in harder test cases.  Give very concise feedback.  In- vestigate the code problem and the provided implementation.  For each failed unit test case, start  analyzing it by saying “The code did not pass this test because...” .  Explain why the current imple-

mentation is not getting the expected output.  Do not provide a revised implementation.  Carefully suggest why there are issues with the code and provide feedback.”

{Test-time Instruction}

\*\*The coding problem:\*\* {Problem}

\*\*Code generated that must be evaluated for correctness and runtime performance\*\* {Code}

\*\*The test results:\*\* {Local test Results})

**b**

| Task | Method | Completion Rate (%) |
| :---- | :---- | :---- |
| LeetCode Hard | Zero-shot Reflexion (1 demonstration, 5 iterations) ![][image83]TextGrad (0 demonstrations, 5 iterations) | 26 31 ± 1.2 **36** ± 1.8 |

**Extended Data Fig. 3 | Code optimization details. (a)** We show the test-time

objective we use for code optimization. **(b)** We report the results for LeetCode Hard using gpt-4o. We report the standard deviation over five random seeds in brackets.

**Article**

**a                                              c**

Example  Multimoda  solution optimization      visua  mathematica   easoning

**Question**: What is the sum of the accuracies of the algorithms fat and acre?

**Solution at initialization:**

To ﬁnd the sum of the accuracies of the algorithms “fat”

and “acre,” we need to accurately interpret the values from the bar chart.

From the chart: The accuracy of “fat” is represented by

the fourth bar from the left, which shows an accuracy of 6. The accuracy of “acre” is represented by the last bar on the right, which shows an accuracy of 4\.

Now, we add these two values together: 6 \+ 4 \= 10.

The correct sum of the accurac.ies of the algorithms “fat” and “acre” is 10.

**Output of the loss function based on multimodal context:**

The existing answer provides a clear and logical approach to solving the problem by interpreting the bar chart and summing the accuracies of the speciﬁed algorithms. However,  there is an error in the interpretation of the values from the chart .

Upon reviewing the bar chart:   The accuracy of “fat” is represented by the ﬁfth bar from the left,

which shows an accuracy of 5 . The accuracy of “acre” is represented by the last bar on the right,

which shows an accuracy of 4\.

**Solution after optimization:**

To ﬁnd the sum of the accuracies of the algorithms “fat” and “acre,” we need to accurately interpret the values from the bar chart.

From the chart: The accuracy of “fat” is represented by  the ﬁfth bar from the left, which shows an

accuracy of 5.  The accuracy of “acre” is represented by the last bar on the right, which shows an

accuracy of 4\.

Now, we add these two values together:  5 \+ 4 \= 9. The correct sum of the accuracies of the algo- rithms “fat” and “acre” is 9 .

|  | Dataset | Method | Accuracy (%) |
| :---- | :---- | :---- | :---- |
|  **Natural Language** **Problem Solving** **Tasks** | Google-proof QA | CoT Reflexion GPT-4o best reported ![][image84]TextGrad | 51.0 51.0 53.6 **55.0** |
|  | MMLU-Machine Learning | CoT Reflexion ![][image85]TextGrad | 85.7 85.7 **88.4** |
|  | MMLU-College Physics | CoT Reflexion ![][image86]TextGrad | 91.2 94.1 **95.1** |
|  **Multimodal** **Problem Solving** **Tasks** | MathVista | CoT Reflexion ![][image87]TextGrad | 63.8 64.0 **66.1** |
|  | ScienceQA-IMG | CoT Reflexion ![][image88]TextGrad | 91.6 90.2 **92.4** |

**b**

Solution Reﬁnemen  Objective

LLM ("Below is a multi-choice question and a prediction.  You are a critical and creative scientist. Your job is to investigate the prediction. Critically go

through reasoning steps, and see if there is a reason why the prediction could be incorrect.

Use the Janusian Process, think about whether alternative answers could be true. Question: { Question}

Answer by the language model:  { Solution } ")

**Extended Data Fig. 4 | Solution optimization details. (a)** Solution optimization

for zero-shot question answering with gpt-4o. TextGrad outperforms the

baselines consistently across different tasks. **(b)** We show the test-time objective

objective we use for solution optimization. **(c)** We show an example where the mistake in the solution is identified in the test-time objective, and later the

updated solution fixes the mistake.

**b**

**a**

| Dataset | Method   Accuracy (%) |  |
| :---- | :---- | :---- |
|  Object Counting | CoT OPRO DSPy ![][image89]TextGrad | 77.8 82.8 84.9 **91.9** |
|  | CoT | 76.7 |
| Word Sorting | OPRO DSPy ![][image90]TextGrad | 77.8 79.8 **80.8** |
|  | CoT | 76.7 |
| GSM8k | OPRO DSPy ![][image91]TextGrad | 77.8 79.8 **80.8** |

Example: TextGrad optimized prompt for gpt-3.5-turbo-0125

**Prompt at initialization (GSM8k Accuracy \= 72.9%):**

You will answer a mathematical reasoning question. Think step by step. The last line of your response should be of the following format: ' Answer: $VALUE' where VALUE is a

numerical value.

**Prompt after optimization(GSM8k Accuracy \= 81.1%):** You will answer a mathematical reasoning question.

Restate the problem in your own words to ensure under- standing.  Break down the problem into smaller steps

explaining each calculation in detail. Verify each ste

and re-check your calculations for accuracy. Use prope

mathematical notation and maintain consistency with the context of the question. Always conclude with the *ﬁnal* answer in the following format:  ' Answer: $VALUE' where VALUE is a numerical value.

**c**

Example: TextGrad-optimized in-context demonstrations for gpt-3.5-turbo-0125

**Demonstrations at initialization:**

**Input:** I have 1 banana, 4 strawberries, 1 apple, 2 peaches, 1 plum, 1 blackberry, and 2 raspberries. How many fruits do I have? Answer: 12

**Input:** I have1 blackberry, 1 peach, 1 nectarine, 1 plum, 1 raspberry, 1 orange, 1 strawberry, 1 banana, 2 apples, and 4 grapes. How many fruits do I have? Answer: 14

**Input:** I have 1 apple, 3 bananas, 1 strawberry, 1 peach, 3 oranges, 1 plum, 1 raspberry, 2 grapes, 1 nectarine, and 1 blackberry. How many fruits do I have? Answer: 15

**Demonstrations after optimization:**

**Input:** I have a banana, four strawberries, an apple, two peaches, a plum, a blackberry, and two raspberries. How many fruits do I have? Ground Truth Response: To ﬁnd the total number of fruits,

we add up the quantities of each type of fruit mentioned in the input.  Banana:  1, Strawberries: 4,

Apple: 1, Peaches: 2, Plum: 1, Blackberry: 1, Raspberries: 2\. Adding these quantities together, we get 1 \+ 4 \+ 1 \+ 2 \+ 1 \+ 1 \+ 2 \= 12\. Answer: 12

**Input:** I have a blackberry, a peach, a nectarine, a plum, a raspberry, an orange, a strawberry, a banana, two apples, and four grapes. How many fruits do I have? Ground Truth Response: To ﬁnd  the total number of fruits, we add up the quantities of each type of fruit mentioned in the input.

Blackberry: 1, Peach: 1, Nectarine: 1, Plum: 1, Raspberry: 1, Orange: 1, Strawberry: 1, Banana: 1, Apples: 2, Grapes: 4\. Adding these quantities together, we get 1 \+ 1 \+ 1 \+ 1 \+ 1 \+ 1 \+ 1 \+ 1 \+ 2 \+ 4 \=

14\. Answer: 14

**Input:** I have an apple, three bananas, a strawberry, a peach, three oranges, a plum, a raspberry, two grapes, a nectarine, and a blackberry.  How many fruits do I have?  Ground Truth Response:

To ﬁnd the total number of fruits, we add up the quantities of each type of fruit mentioned in the

input. Apple: 1, Bananas: 3, Strawberry: 1, Peach: 1, Oranges: 3, Plum:  1, Raspberry: 1, Grapes: 2, Nectarine: 1, Blackberry: 1\. Adding these quantities together, we get 1 \+ 3 \+ 1 \+ 1 \+ 3 \+ 1 \+ 1 \+ 2 \+ 1 \+ 1 \= 15\. Answer: 15

**Extended Data Fig. 5 | Prompt optimization details. (a)** With TextGrad, we

optimize a system prompt for gpt-3.5-turbo using gpt-4oas the gradient engine that provides the feedback during backpropagation. Supplementary Table 1

includes ablations with instruction-only and demonstration-only optimization with TextGrad. **(b)** We show an example of an optimized instruction for GSM8k.

**(c)** We show an example of optimized in-context demonstrations for GSM8k.

**Article**

|  a |  |  |  |  |  |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Target** | **Method** | **Mean dose \[Gy\]   Min dose \[Gy\]    Max dose \[Gy\]   D95  \[Gy\]** |  |  |  |
|  PTV | Clinical Goal Bayesian Optimization Radiation Oncologist ![][image92]TextGrad | 70.20 \+4.84 (1.65) \+1.97 (0.36) **\+0.51** (0.09) | ≈ 70.20 −9.33 (2.38) −8.88 (2.31) **−8.48** (2.38) | ≈ 70.20 \+17.42 (2.88) \+4.66 (0.82) **\+3.63** (0.87) | 70.20 \+0.00 (0.07) −0.10 (0.15) **\+0.00** (0.00) |
| **b** |  |  |  |  |  |
| **Target** | **Method** | **Mean dose \[Gy\]** ![][image93]              **D5** ![][image94]                                                    **D50** ![][image95] |  |  |  |
|  Rectum | Bayesian Optimization Radiation Oncologist ![][image96]TextGrad | 24.47 (9.30) 23.88 (6.45) **17.18** (4.20) | 62.01 (10.79) 64.26 (10.00) **58.82** (18.81) |  | 18.74 (12.12) 20.04 (5.50) **9.54** (0.70) |
|  Bladder | Bayesian Optimization Radiation Oncologist ![][image97]TextGrad | 26.15 (5.64) 22.39 (5.55) **20.92** (0.79) | **63.57** (8.82) 67.81 (6.44) 65.96 (6.96) |  | 22.19 (5.71) 14.78 (8.42) **14.11** (3.17) |
| **Extended Data Fig. 6 | Treatment planning details. (a)** We display the several dose metricsofthe PTV target for all the clinical and TextGrad optimized plans, including the mean and minimum doses, as well asthe *D*95. For all the metrics,  we include the average deviations from the clinical goal across five plans and the standard deviation in brackets. Values in bold represent the best for each |  |  | PTV target. **(b)** We show mean dose to capture OAR sparing. Lower values demonstrate better OAR sparing which is desirable, as this number indicates organs at risk, which should not get more than dosage than what is listed in the clinical guidelines. For all the metrics, we include the average mean dose across five plans and the standard deviation in brackets. |  |  |

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABdUlEQVR4XpWMQUvCcBjG/+fOHbp2DLx1j6gPEF2iex+gvoCQh6irEHVyEt4DU6hDJzfdpuCYc6TkhgpTUHSCinp62vvChmIXB78/e5/3+b0imUy+BsR3JC2CZ0/s+AXOgZhOp1gul1itVhFZx8PZh47znI682+N9yGKxwGg0gphMJpjNZpjP58xJVmOupRyu0vloDvfUHQwGEPSMx2P4vs+Ft7oL//h0gxfjl3fUoa7neRC9Xo8vmMUbxL+eMRwOt0TKSEyoNnc7nQ5Et9sFyVBjKOqP6Pf7WyJlT5rNMnVd14VotVpot9ssymqCr6lH+7Bih5FI2b1SY5G6jUYDotlsgmRVuWPZcRwWKSNIpIykB8XkrF6vQ9i2zRfoAImacsvzOp+FBIvUodk0TQjDMFCr1WBZFkPyf4R76lYqFQh6qtUq6EDIt/qOpnyBn8Jl8J/f2FFX0zSIUqkEXddRLpcjaF5nfUfIsgwhSRIymcxOpFIp/AE+/kAs8QNhmQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAECAYAAABGM/VAAAAAKElEQVR4XmMw8ZrFAMVPYGyYgBEQf0cX/AUV/ATEIiCBd0BcA1MFwgCE5xxPZr9kIQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAJCAYAAAAGuM1UAAAAmUlEQVR4XmPQUFb3BuL/DEgAyI9BF4MDkAQU3wLiS1AMExNDVw8HSAq/APFvqJgJEG9HVwuS2AvEd4HYDYj/AbEqEPchyYMM2g3EjMhOegVV/AxJ7AUQP4VisBjMhI1AfA2IfaCaZIG4Cc2G9TA+TPAMEP8F4h8gGipmB8Sr0RXeg5rgjSQGsgUkJoKsFibpDJJEEwtGF4MBAOBYS0d1866ZAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAECAYAAABGM/VAAAAAKUlEQVR4XmMw8ZrFgIQVQTSyQC8Qf4cJ7gHiP0D8E4i/oqsEYTUgPgAAYWocEpJ7C4wAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAYklEQVR4XmNgQAKq8sqrgfg/EHshi8MBUEIYqkAUym8G8dHVgSQwBIFiYkB8EV1wLooAFKAYAOSIA3EdkjwcYNiEIQAFGOJAgSNAvApNDKtmkEQ+SBIJM+NUjA6ACplIVgwAKn4hCFkmnGIAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAJCAYAAAAGuM1UAAAAlElEQVR4XmPQUFYPAuL/DEgAyE9CF4MDkAQujK4WBQAV9AHxDSD+AcRfoGJTQS5AVwuSaAHiF0BsAzMZSNtBaW4gfgXErTDFMCesg9Lr0cRAeCtMDKSBBYifA/FfIBaESmgD8QIkF4DEXsD4MMGdUImDQHwOKmYOxJvRFX6FKgxFEoM5SRJZLUxyJUgSTWw3uhgMAACtSUnkV+GAzgAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAJCAYAAAAGuM1UAAAAm0lEQVR4XmPQUFa3AOL/DEgAyHdDF4MDkAQezIyuHg6AkouB+BkQ/wPiv1CxKiDORVcLkmgF4itAHAI1mQWIfaFyIPZXkBqYYpj1m9FoEF4PxOugNFgMpIERiF8A8Xcg1oBKmAPxBiQXgMSeAzETTAwkCHISSOIyED+BiqkC8Um4IqjgfajCPCQxmJM0kNXCJNtAkmhis9DFYAAAUItLmnAAgUYAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAqCAYAAAAAsVPKAAAAuklEQVR4Xu3ZoQoCURBA0SdoEJOoTTDqB9g2i9kqmEQQMSnq7ztT7yfceQem3LawsG/ftFbPg8FsFLNnNFsw2P0YzD4MZtOYDaPZi8HuzWC2jpkxmn0ZzFYxc0azUt/ddGcwezKY7WImjGblDhp5lCzjwGB3YTAbGOxuDGZnBrsjg9mJwa7UA48Z7PomwezKYFbqCJn6JsGs1A9+Xt3kFU4ZpS7nUqlNwrL1TYJXqVc55dqzjG3rmwS3P4c+DTQutruiAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAJCAYAAADgkQYQAAAAUUlEQVR4XmNgwAJU5ZUvALENujgcACX/o4thAJAiIPZD4kshy8MEQYpA2BLK/4iuQBRJERijK7gElXiPIgEDxCiAWYFdAQgAJdcCsTG6ODIAADzWHYX1W0/rAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAdCAYAAABWk2cPAAACIklEQVR4Xq2Wz0tiURTH/VMGakqdRVGz0m3UwBQVRNCqWcxSCloE0TJaGDQwi1YVQZGWWVKk049NSEFk1jASzhth/JWL/obK2zkvtec57/nes3vgs5B73vfzzuVe1eGwUc6W9jVA6PCT9r6rIHD7c0d3OZfLiVKpZIjyVxGf2tz4Aos0w3K5Wp0nxWKRhVshk8nYl+OW0aBmgBcv02zdkiWsgnnUUVeyhVVg4ifqUgsWjmmzTGCgaep0mB2aQqEgPB6P8Hq9psTjcfb8TTJZv83wFkHaRIUf2jvF+MSsiP55MGX38j/LQMDzvSbt7ugq0wYq9Ad/W5YiNKcifZvW6OJrhTKkR7+OXqVgX6GLVbTC+Y2E2LvKif6BIUP8y6GGUgR8Y4bXJBQKiUgir7KfvGeTmEHzNFLRUFoNmJyZY1PpsbCyI0+K4Ik0Q+qkywfnbCo9di7+yZMubZ8wgR7BszvL0h90gUo3Tm/F4PCIKRE43RakX9Vrk06n2aJWurgWYVPpsX5601C6FQi+fTm421ysIRwO1wKageYhdd9I8MFPGxC8ozTMCquxK5ZVkfbWpFiKorCmVColfD6fbbLZLMuKRWP8x9xoWhnk83mcso861cL/NPQBGbg/uh6oq66M7m2zwCDP1KFb0PhIH7YLbqnphLRg4qnrxDULswIeGnZS7RQ8/A23PHoYZeFaApsB9R4CPTTjXQWBo5Vgyhfa26heAPEJL/k6eAoJAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAALCAYAAABGbhwYAAAAcUlEQVR4XmNgAAJjtxnuQPwfiGeB+DgBVBEIqyOxkfEvZEX8WBSAMcw0JqjALij9CYizkTATstUeyLoxTEMGSJJ1QPwLyt6Prg6k8BRU8jFO02AAquAblGZGl4cDoGQgVNFvdDkMgNdKZABUyIEuBgIAn45oE0emTKEAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAADCAYAAACuyE5IAAAAE0lEQVR4XmNgYGD4TwATBu/xYQD2AQ9xRfuGqQAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAYAAAC09K7GAAAAJUlEQVR4XmNgyLH4D8TtQLwPiB8xAIm1QCwJxAFAXAISkGRAAgBhKQqOX8vprgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAANCAYAAAC+ct6XAAAAwklEQVR4XmNgIB58BOL/QHwADR8E4jNA/BYqD8NUBSADT6ALYgF3gJgTXZASIMwAsXwHugQWkIcuQCk4zACxnA9dgh6AKvH4AQ2DEslrIDZBVoQFkGJ5EhDPRuI/RmKTBUAW70MXxAFgjowCYjUQYw8OrAlViA98RxfAA75AaWJDCSdYhy5AAKwB4nsMFFoMSgOkAAMonQDEYkjiJAFiXfwVjX8eiJnRxIgG19EF0IAbEH9iIN5xRIFLDKhlMSGMFwAAtw093qSlJ98AAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAHCAYAAADAp4fuAAAAPElEQVR4XmNggAB2IC4A4p1AzAgVY/gPY6CxGZ4BsT1MkA+Ie6ASLDBBZOWGQFyGLghncwLxYyD+CRMAAFb+DHo9g1LLAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAHCAYAAADAp4fuAAAAPElEQVR4XmNggAANIA4B4gdQPhj8x8bmAOK3QCwCE3QCYiOoZDAQ/wMxkLV2A7E8uiCcrQvEZ4H4OUwAAHEMDLDB6L8QAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAANCAYAAAB2HjRBAAAAjElEQVR4XmNgIB1UAnEMuiAusBlKNyKJMQGxEhIfJ/gPxJxAzIxFnCAAKbqLLsiApPkxEHMgSSCDJCCWYUA4HwZa0PhEAxQnMyJzSAU/gXg5Ev8fEhsv+AulnwKxFpL4MiQ2TgByPysUr0cTJwhAip5A2Q1o4kQBkMKvSHxQAKoi8YkG1UDsiy6IDQAAoD0Y/v5gYdQAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAGCAYAAADkOT91AAAAGklEQVR4XmOYwCfzHwm/YYAyJBhggHwBuKEA6kMXz0kEAh4AAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAGCAYAAAAooAWeAAAAfUlEQVR4XmNgQAMT+GT6gZgXXZwsADToCBAXAPE2EA0VswLim+hqYQAodw5dDBkA5f8A8X8Q4z9U4B2MDeXD2egAKPcVXQwGgHIuUBpiOBD/gwr8QlJEluEwADNcFGoBCDNCJcSB+Bu6BhgAyp1HF0MHWB03ARKhTuji5AAANhYxVMDOChUAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAGCAYAAAAooAWeAAAAdklEQVR4XmNgQAc5Fi+AmBFdmDyQY/ETiO2B+B0QK0HF+oC4FU0lAuRY3EAXQgE5Fv/BGMpgBOJ1YFcjK8AF8MtNh9Jww28BMQsQr0JShM8A3HIwADU8GGoBQkOORQwQr0RSigpyLK6jC2EArA6ARKgkujA5AAA3Ry9iwPKDJQAAAABJRU5ErkJggg==>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAICAYAAAC/K3xHAAAA90lEQVR4XmPQNQ3/r6GsPiRxvKrWfwYQwzxs4X+zuKX/zdNX/TfLX/ffrHzTf7O6rf9NW3f9N+3d999kxuH/JgtO/DdZcfq/yYYL/013X/lvv/Pgf89t8//7ben8H7yp8H/Exuj/SZsj/2dvDf9fviPkf/3uoP/tewP+Tz0U+H/h8aD/a08H/997KfT/yeth/6/eDf//8FHE/1ePI/6/vhXy//VZv/9vDnv9f7PT7f+b9U7/3yy1//92jvX/t/3m/9+1mfx/V2P0/22J0f83Wcb/X8cY/3/lZ/L/hZ0J2BOjHhj1wKgHhroH9G0yMIqnoYIL1bT/AwD3XHRM8E4b9AAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAZCAYAAACRiGY9AAACu0lEQVR4Xu1XS8hOURRdMSAmysQjyqPklQw8IvqLMvAaGUgGxAgpZaRE+cNMGTBhYKQUUVKSxETyLq9ShEjySnm/1nL2dvd33O/7v5tv4NddtTr77vNcZ59z775Ajd6HoeR+8gK5w3wTyLsZq+BS7qiATbnjb3Ga7EtOJU9mdVXwOnf0gO+5oxNYQB4hu8hpKBel6B0Pz76QL+Rl8gc5yHwuarWVs4OtdteQ+s8gv5rPx9tp5Uzza/yH5ttO7iVfooeNmEceMPsjOQdJlAZ0frN6H2gkuQspqppI6CJXmd1KlDZHGEweNTsu0EVpXscVcjjSXGvN14cc87tFhifBPoS022WREs6SA8mbwXcbaVG6l+2I+oSifStRiqBjOdLYEjXOfBLl9h/QLqiB4IM3E9WPPIUUUWEJudFsn1hwUeus7EYhaoCVo9FaVIyUNlEvtLZFqVIDaGc0kZAfvziBjuJis/tbnc79QfK9+V2U6p6TW1CIkoAP5DkU0ZAvv1OTUdy1O+ZrW1SNGjU6g93/IWv0Sijd0Xvf0yEhfqPi170K/LsSx80xESnjEFYipW0dxWZyhNlPg389UlpTFfFj2QzzkcR0HGPJM0hJpoQJUZSS3BNmL0SKntIdZRIP0JjmPCPfIGUX6pNHSn1vIfUZb8+ihHmktA75lEl4liJcR+r3inwR/KXwBPURuc1sX/BjpAl0RIWYAE+xchK5lRxCXi2qcQ+Noi6Sw4pqnEdjpFxU3IQ95FKzPwd/qyP9C75gJaWzzI6RilgR7HjvJGAZuSbUK9mNopTz5SgTpfEco1AkvtpgR4+iFG4hTtpMlBbumG6l+uv+qIy//vfRKErHJ0bqLcpFxQXvIxeZXUnUMfJd5mtH1AakXdXEDv246dzrqJS9/Q6jiKxDWb0EuSjhBlK7ud4IFUXV+FfxE5RP46wUODIPAAAAAElFTkSuQmCC>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIsAAAAUCAYAAAC9Kc5pAAACAklEQVR4Xu3ay2tTQRTH8dRarQ+kioq6UMGKUBV8nIWIYAVfBES3raVK6d4noigqPkAEF+JGgl250j9Au6oILnQhLkRwIYJCN127shT6nR4LzeSUJCY3vbk5A582nd+EBvLj5s6QXG7OkHwh6xahG0cxjPt4jEd4iAe4h7u4g9u4hRu4jmu4isu4hAsYwmkcwV6sldL/mwlFIw6bzEqcwVOM4AUKom94P3ZimfG8JKzGbpzEOQyIvoY+HMNW4zmp14xlOYxneI1XeIlBtBtr02wV9osW/BROYJvo1S9emwppLstZvMUb3DTyrFuCPaIfbwexxljTUGkqy3mMYRSHjNzlC5tEi3MAPdLgq+lClyXcIH4QvYGMM1feYuwTvQKtN/K6WoiyXMQn0d1GnLnarBO9kd8h+jEW5zVpVFl24YvoVjXOXDLasEX0pjnsFOO8akmWZQO+4SM6jNw11kZsxgojq0gSZQlnCt+x3chcOoSDw1CecPWJs3nVqyxL8VP0Dj3OXLqFc52wLe8ysiK1lqUXv3HcyFzzCburcFgYfsfZf5flCcax3MhcNoT3tug0udqyXMEPY95lV9iCz5Sm0rK8x4Qx71pHW7myvMNXY961oJKJf8L3Nv4Y866Fzf1jEs9FvwhUstC52QdTomX5Gy9wblb48Rm/RHc6nTOTPnwYYxr7T0/1ybVvHgAAAABJRU5ErkJggg==>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAANCAYAAACgu+4kAAAAyElEQVR4XmNgAAINZfWXQNwIYiMDoJgWEP9HF0cBQAU3QIqgOAgNw8RxGwKU3IOkkA+IeZAwYQNgAKgoGYhnQjX8A+J+IH6JJE+UIX/w2QoVF0cXZEfWBMV/gfgdEP+G8mOQ1D9FUicKM5VsDDLAHUngOBI7Dol9D8kFyLFTCBOHSR5FksQWBqD0shJdHAw0INH2CsmAbCBuQZJ/CMTCyHpgEsVImgqQxCWQxDFcAwcakDiHK4TyYZiwASCASxFQbDdUTgFdDgYADAqSv4RnrrQAAAAASUVORK5CYII=>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAANCAYAAACgu+4kAAAAzElEQVR4XmNgAAINZfX/QKwMYiMDoFgHVI4ZXQ4OoApgOAgJhyLLoeuDAzQDeJCwIFEGwABQ0XsgngHEf4H4HRAvAuIQJHmiDEF2zU8c8o7ogplQid9A/AqIfwHxXTTDViGp34kk7opuK8kYZMBzJIEQJPZ5JPY/JBcoIokfhYnDJMGKgfgjEG9HkYTIHwDi2+jiYACUmAzE8UimgwxhR5IHiVkj64FJgGyEaRJGEq9GEscdhciKoIbBMLI4XgPUoYo2YJGDGRCOLgcDAEdgltJWzN4tAAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAAA10lEQVR4XmNgGAhgAcRdQFwHxJ+gYlZA3AxXQQCcAmImIH4AxPyoUpjgLBDnAvF5KP8fkhxW8B5KCwPxfyj7L5TGCnSQ2GxAfBjKvoIkjgFakdjfoDTIT5FI4hiAiwGiGOQHQaiYAwPEmTBM0H+DCNwlAw8QuAfEXgxkBq8uEOegCxICoJTdzYBqI1bbVwPxYiB2BuKTQGwIxJ5A3ADEb4CYA64SCexFYn9BYoNsUEbiowApKM3DAMmEILABiDOAuBjKxwCgRPkSiCdD+dFAzAtl/4LSdAYAP00ymVUQuxkAAAAASUVORK5CYII=>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAZCAYAAADqrKTxAAAA2ElEQVR4XmNgGAggDsT/gPg/EBejyREFfkHpWUDsgSyBD+wDYiYg/osugQ5ACpyAeDcQf4eKgZyKE3ABsSAS/wOUfo0khgFqkdirgFgCym5FEscAdkAsCcRZQLwUKqYAxDlAHIuEMYAZusAoIBHUAfFqdEF8QBtKMzMQkeawgffoArgAGxDfAGJXIP7MAEnlFlA5USA+BmWjgAtQ2oYB4byJUBqWv1BAJhJbGYgboex7QKwJxNUIaQTYAaVZGFD9A8r+eP0HynCT0cRgfiMJ7EUXIASweh4EAFN7JXG+n8YxAAAAAElFTkSuQmCC>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAKCAYAAACJxx+AAAAAPklEQVR4XmNgIBGkoAsgg1IgjkUXhAEmIJ4MZR9GloCBG0js/UhsOPiPhiehShMAoUAciQMTB74yYLoBhhkAagcSDhHd9NUAAAAASUVORK5CYII=>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAKCAYAAACJxx+AAAAAN0lEQVR4XmNgIBGkoAsggytALIouCANMQDwZyj6MLAEDN5DY+5HYVAKhQByJAxMHvgLxfxyYAQDXkA2XepKEvAAAAABJRU5ErkJggg==>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAAA0ElEQVR4XmNgGChwCIiDgfgflM8ExBcQ0vhBFhDrAbE1EK9Ek8MJLIDYFYj3ALEOmhwKuA3ENUCcAsTngZgbiP+gqMACqpHYMD/9RxLDAMIMEE+DwHIgVoKyb0FpnABk+hcgjkUSA9mEjEcB1cE9IPZiQEQuSUAXiHPQBXGByQyQRCoJxHlA/AaI2YCYnQES8RgAlDiZoWyQ8wSBmAeIzwHxX5gidJCExEb2E17/SUHpVCAOg7JBtoNsw5l7QQnyJQPEXyAQDcS8UPYvKE1nAADYHiciqQwAawAAAABJRU5ErkJggg==>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAZCAYAAAASTF8GAAAAUElEQVR4XmNgoAQsQxcAgTYg/oMuCALRDAMoUQjEf9EFQeA4EP9HFxwuIBaI3wDxPwaIJ1E8ihxOP4CYCZvEV4ISIAJZ4hdMQo4B1UKQI7QAFnoacD9nCc0AAAAASUVORK5CYII=>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAYCAYAAAA20uedAAAAjUlEQVR4XmNgoBQwA/E/ID4NxDPQ5Bj+AzE7EhsOuIH4PRIfJMkE4xwFYi+EHKpOkF0gAWQMB8gcByA+hsRHkfwIxGxIfLCxICCNxIYDfgaI7tvoEsMU/AXicCC+B8SXkCWuA/E+JD5G4Ach8d8BsSKMA5KMQ8iBJSVgnAQG1EBHMRYETBkgCn4wQJMJABFnJE2cEpHqAAAAAElFTkSuQmCC>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAfCAYAAAAx6zerAAABEElEQVR4XmNgGPrgN7oALtANxUSB/+gCuAAzEL9BF8QFdgNxMLogLoDLCX+B2BmInwGxLkiAhwG74vtAvAnK9gbiJyBGNAN2xSAxJii7FIg/ghjroBLo4B8S+yEQXwUxzjFgKrYC4gtArA7FII2JIIkrDJiKe4HYH4kPd9IhKAcZzARiYSgbFAdfYBIlDJiK2YG4BcpGdjuDPAOmYhDQAOIwdEEQQA4mguAXENugC+ICSUB8C10QH8DmbpzgIBDvQBfEB1CCiWoA2c0vkNhY/QISNEZiS0DZj6E0CrjEAEmGlkCsBcTngdiMAWEACmBjgJgIKztA7G8IaUwACo1GKPszAyT54gS7kNgODJDEBAcA1tY9F7qyl9cAAAAASUVORK5CYII=>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAfCAYAAAABZyaKAAABmUlEQVR4Xu2VvyuGURTHTxSDFAYGkywssigWk3+AIotBSUQpZWOxGPwJLBSyKUkGTBSTLAZF+TGQn0khv77f9znXc9/TfdSjZ/StT+ec733e897nvscl8q8sdAAqrOm0Bi6tmaBPa/hqBi1e3Q/WvdrpCIxZkzoHF2BJ61tvzapKAru511gNZjXf0ZikL2u8a9wGhWBQ42+6Bh2+8Qo6wbTW3CoZl2h3IS2AY2umVTf4sGZaNUjgXNKqRjJoUisZNGmUQJMBjW8aHzTyAEMalsDAlYNDiQavFJyqP/rzRL42wJ01tzS+gHmJtuoI6QlM+kYXKNCcv/2VVyc1oZ831RMai8AqONO6Dsxo7qtEEprTXNG8WKJDW4yX87Qp0ev8WXyF4C7S6AbUWzMz8bpr0pyHxquBegTtYBeUqcfr8xlUihm2ZYmbUK5J6N19j+PQ5oo5CTfh9PJDQ94aa647ptxCUhMnjjevQyq0u5xGJL6geyW+c3s0toJ9zXkOPCuK59SneU78Bj7APz73bXuan7iHVJxo+vy/JN+4+mNVELaYcwAAAABJRU5ErkJggg==>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAYCAYAAADZEIyjAAAAkUlEQVR4XmNgoAQ8AeLTQPwbWfA+EM+Bsu8CsROIwQ3E/2EqgOA2EE8EMY4C8Uckib9AXABi/GOA6EDG0iAJEIMJqtoBygcDZPNBRs6GcUBGYWMz8DNAdIEsHcpgGwPCYypAfAFJjuEPEB9gwOJJFgaI79nQJaSA+DkQH0cW5AHiO1A2yK4lMAnkELVjgBgpDACu8CXPfAFeNQAAAABJRU5ErkJggg==>

[image36]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAeCAYAAAD6t+QOAAABDElEQVR4XmNgoBeoBGJeJP51IP6FxIeDJUDcBsQTkPggEAyl4WAWlFYA4jVQ9isojQH+Q2kHIA6Esv2hNAb4DaU/Q2lkd4MAMzInDYj/IgkeBuIjDBAbX8IU0RaArCIWkwdAHtwMxFpQPjsQfwPin3AVWABIAQjAFEXCJJABLKxBNoDAHCD2BOLzUD4cqACxAJQNSjggPgeU/w9Kw0E3EvsLEFtD2UwMWNIIyFRFIH6HJKbLQMBzyMAciEXRBXEBUCTA/EEQgBRzowuCQBQDIkph4YpTMSze9RgQQYVTMQiAgukmA0IjTsWgBA5K+KDgw6tYGCoBAuiKBaFsFACSAEUAqNxAVqwPVzF4AAC+9EM2ufpRhwAAAABJRU5ErkJggg==>

[image37]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAXCAYAAAAoRj52AAAASUlEQVR4XmNggIAmIHYA4uNQPhhIAPFXIE4A4lVA/AZZ8gcS+zcSG7fETyT2HxiDGYnDBMR/YRJDA/xHFyAIcHoQp1H0kCATAACEChNdrajdqAAAAABJRU5ErkJggg==>

[image38]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAZCAYAAAASTF8GAAAAXklEQVR4XmNgoB/4hy4AAseA+D8QfwViZjQ5sARWQJ4EE7ogCGxngEiyoEsMJwAKejEoDQd3gDgNyq4DYheYBCg4fiLhdpgESHskjIMM5jCgBn0KEpuhhgEieRXEAQCSkhehZAZvpwAAAABJRU5ErkJggg==>

[image39]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAXCAYAAADHhFVIAAAAVElEQVR4XmNggAA2IK6FskEgE4nNIAjET5H4/5HYDMwMlEg+Q+KjSMIEQoB4P5Q9HMFHdAEYAEmAPA3CTGhyDBpQCayAfEk1BjySIABzkBK6BA0AALzIFkVC7CtiAAAAAElFTkSuQmCC>

[image40]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAZCAYAAAD5e+QFAAAAXUlEQVR4XmNgoBoQQhfIAuLV6IJ/gfgXEH9FFnzCgEUl8YLXgfgxuiAzEP9jgFg45ADI4XJA/A2ImUACFxggPgIBJyjN8B/GQAYgrRhgFhC/R+ILwhiPGCDGHABxABMSFdrVA64kAAAAAElFTkSuQmCC>

[image41]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAZCAYAAAASTF8GAAAAcElEQVR4XmNgIBesQhcAASYg/oUuCAM/0AVgACTxHwmjSMDAHyQ2iRIgVyGrArFBYgzSQPwPSQJkuRQSf1iAv0AsygAJGg+Y4HIg5oCybaCKwADE+ImGwQAUrSwwDjIIY0AN3R1IbIbjDJBQfQsTAACRzR901DeM9gAAAABJRU5ErkJggg==>

[image42]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAXCAYAAADDcYV1AAAAWklEQVR4XmNgQAAWIJ4DxHIwga9AfBqIlwGxEEzwEBBLwTjIgpLogscYsKi8D8Ta6IIDC16gC1gB8Wp0wT9AzIUu+B9dAASIF/wLxGLogl4MkFDHAK/QBYgDADfpD7o6/vQaAAAAAElFTkSuQmCC>

[image43]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAXCAYAAADHhFVIAAAAdUlEQVR4XmNggAAmKF0DxNuhbDj4C8QfgdgFiF8BcR2y5AsgZoaytYB4NZIcfkmQUTBJXSDegCTH8B2IuaFsJyA+hSQ3xMBPdAFkkIEugA62oAsgA1DM4AT/0QWQwT90ARgApYSD6IIwMIkBkVQwwC90AcoAAAsUE4idF5stAAAAAElFTkSuQmCC>

[image44]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAZCAYAAADjRwSLAAAAd0lEQVR4XmNgGBDAhC6ADfxDF0AHz4D4PxB/RZdABwRNAgHqKQK5iSC4zUCkwlFAHEAOTBibB4htkcQZzgCxMZQNU+QPpeFAGoivArElEO8AYikgfoGiAgpAJvyGsr8A8UkkOTgAJZlGKBukQQ5JDg52IbHfwRgAnOoZIOYVCSgAAAAASUVORK5CYII=>

[image45]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAZCAYAAAD9jjQ4AAAAeElEQVR4XmNgoClgQmJ7AvFtZIl/CDlU8A2I/wPxVyj/OBAfg0lqMaDqVAbi3TAOuqQcA7GSIA7IThgAKT6NxAc7/S+UXQ/E35HkRgwoA+KXSHyMKEQOwhQkNhiAJA2BWA9dAgRAqkHG/UGXgAGQ7ix0QRi4jswBAKYtG8Kh+ZWhAAAAAElFTkSuQmCC>

[image46]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAZCAYAAADjRwSLAAAAnklEQVR4XmNgoDv4AMTq6IJkgX9ALAxlc0H54lA6BKboARDzQNkgCQMomwWI/0PZDNcYEIpAgkwwCSgfDNAVMcMkoHwweALEYlD2LSA+B2WbAvFvKJvhLxCHwzhA0MUAMeEMktgoIAGAQrofiL0YIJGMFfxCYoNiAAPwMkAkfiJhDMAOxN/RBbEBUITWQtkCyBLIQIgB4mCQ4jQ0OQYAsUIik4A2JQwAAAAASUVORK5CYII=>

[image47]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIYAAAAQCAYAAADTRgfPAAAEHElEQVR4Xu2YXchWRRDHh4yUSswkC6KMEo0QMQIjQjQzMAkkQhEtEBJRxIoQhEIruqgLQQoCBUkyhLpRuulCzQRFMb+Cog/LIBGjDz/KSrO05ufOvGfOvM8rz5uPmnZ+8OfMzu7Z3XPO7Ow+j0h7nFbdZ/Z1qmtDXcP/mB9VP5v9cKxouHx5M5XfU92UfI9IyRqwJ1Z0gCtUK7KzoWM8pDqVne2SbzysGmP2A3Z9Q/WU6nMrdwom/nd2XmL0y47/ECy8/H3bZqLq7VD+QfWc2V8FP1ljfChfLvyVHb2AM9e53H8hOKf5xVVLAFxp9rHg/yPYPUGEHlGNktLn3ebnjPKo6nEpL9PZItXYrDzshXZFq6wuc1T1imqsVNsc7FetVr2u+jP46etVu7rgF7N/U31jvmdU36rmqk6a732pj+P2CbO5f2RVfYa9Up5/k2qz+RhridnTVdeY/bvqadViawP+Po7bFc0Otrd7wuyDUs6D2HwDJwYGc1qqmiT159mpuk1aZBc662t2vKG39FfdYvY+1Tqz/SEyt0u9DnuY2W+pDoW6CA+SbV50fAncu91sXtiaUJfHjOS6wWbT9yYp4/kvsyfNn7lTqqBisXift5rdR0owOb8Gm3oPmDiX3VJ/buquDnaE8kCzfX5xTkAQTFBNU00J/hoEw1CzmTQrgQf6t7BSGHiDlYlgJvtFV4vCzeZ3YmAsk3LegZlWx+qByXaNEAifhfLLUgU5q6mdwLhXyj1kRxcvD8iitH3NytBTYGyU8vyxH2ebdP+Qjmc1FhjEdluld4FBhgafX6s5kXWB9jx3t78iqPDtI2cM/xiZT6Tcd1XwcV6ZZzbnkw9CHTwr5QzjtBsYmXeyQ/lU6u1XSglwaDcw7krlCCn4fqm/q54CY630/N74OD9Jfa55Pq0Cg+zXm8C40Wyf39nmBPTFfaPdEVMdMPEBZl+v+jDUZUjRZBiHtr6aSY8fmc0+6LgPhkv3l3KP2e9KPd1GaDfE7Kl29QdzCPARZvMRfFuDPOagVPbMdoeUYGElfW2+HVItHlZlXkjAYqEfT+ez7Moen7cJzlx5Pr59Rf/Hqi9Dmbobgu2Q+eNZwW2f0yIrMy7zIrgdsq6PfSZ9La/q5DupDkgvSfnXk5VOO5eTAwMYnMMbK55DFRDt+PNLfN78DvYLZh+Q7u2dx6TqL6Y/zknso/iHBD/leF6JY7LlUI4ZhUM3vnFWzvOgbr3Zvk362cphUeHnwzDHB1Xfh/oXpdrz2XJpO0fKWDwf4PMtnYUWzwjU+dnAx+G6q6tFNYdYpn98HGSBRUEZ8ZdEFzkVkn49DfmhaIZqfpDTKjAaLjzx43cEfhpGFqRyTEmtYLX4fttwcfAtIp71zjsM6Ptkxg+fHY/WhoaGhoZLhX8AC5hEoywqZxEAAAAASUVORK5CYII=>

[image48]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALsAAAAYCAYAAACiGIwqAAACd0lEQVR4Xu3bSahPYRjH8WsmJEPIUOZ5ymNBIaEMJaKLhWEhGbJSlDFDyJQxCxYiLA0b0860sCEpCxliY2Nnoax8H+//ivd9uP97+0/n3Gfxuf3v73lO93Z7z3nf99xz6mTuhTpXtK4YjplYirXYiK3Yhd3Yg73YV7AfB3AQh3AYR3AUx3AcJ3ASp3AaZ3AW5wt1PX6HhJ83Bb0k/d1cI5KgBWmFIViEddggYeBuwmZswXLMwhj0iI6vtn6YjjUSTqgruIpruI4buCThJFmMbtHxLU4S5IhehcdhBVZhtYSBsQCj0dE4Js9aYyUu4hZu446E2Wia0Z87SZAxbTFWwpJiGZZImOb7GL3u/3Sm2I77eICHuCnh7xr3ZlIS1Bi9Gg3DfAlX5HkS1sy6BIl7XfnpRUVPhid4ivVGT81KgirTgTwbczAR7YweV1tG4jKeS5gNRhg9NSEJKkgH8njMkLDR6mT0uOzRWVfvIr3AXfQ2eqoiCcqoDSZhqoQNoi5R4h6XP3oj4CVeY5tRr5gkKLEumCxh0+hLEqfq8UbCrdK4VlZJUAKDJay3hxo15/6kG9y3WGjUSi4Jmqm/hFuAeiWPa84V4xE+SNi/xbWSSIIm6IlR6GvUnGsu3eDqGv8TJhj1ZkuCRujSZICRO1cuzyRsbuO8yZLAoP9ZG4QORs25SrmHj0ZetCQo0CXKQHQ2as5Vkz7g9sXIGxUH+vCUP1fisuAcvhr5PzV86C4+yF02fcZjI0/oFx3oScG5jHmHnUb+WxI4l2H6gs03I/8lCZzLgfcSXon8K4+bnMuT72jf8H1cdC5v9DXEV/o5LjiXVz/iwLnc+gm04rb0zjjeMAAAAABJRU5ErkJggg==>

[image49]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAACCAYAAAA5Ht7JAAAAGUlEQVR4XmNgYGD4D8XogG7iBBWgAaqKAwDpsxfp9AQ9GQAAAABJRU5ErkJggg==>

[image50]: <data:image/png;base64,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>

[image51]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAN4AAABBCAYAAAC+VxohAAADZklEQVR4Xu3cPWgUURTF8SRq/IwRv8VoqYXts0+hRjsrG+2jla1a2AoWEmwUIiikEwtBEDvRwkZQiYWV2FikELSRIILxvuxEZk/eZCcwM3dk/j84xebdN/s2zDHZGDI0hM4KU/cPtSQHsxyw7M+yz7I3yxY9O/DfShSgrZmQxzv0tQCoiRVua6KUK9mu8wBqYGXblShgTPxWdYPOA6iYFS1md6KEMTt1HkANrGyjofcDGy1hzJjOA6hB/KqXKGBMLOcmnQdQk1D8HjH+t8ZGnQdQsdB7jxgLpyWMGdd5ADUIa79H3KzzAGpgZRtLFDAmlnNU5wFUzIo2Enq/0qYl3KazAGpkpdsTy6cfB1AjK904xQMaRvEABxQPcEDxAAcUD3BA8QAHFA9wQPEwkN0gfyxLubzTmTYIvT/18DI743VdbxOKh0J2Y0xJ4fqi8570bLnM6mwb2Lm+ZOd7rGvouMRNrPmkezwkzqV5pnu82FkWE+eLWdRZdFDixkhG9zUt9L69XHUuje7zoGdK5KruQcckboqinNW9TUqcpygvdG+T7PnPJc60KroPNdJPPnHLL8uCZd7yyvLU8tByx3LNMm25YDltCZajoeQfQso9x6A80r3okMQNUZQTurdJifOslSuWG5bbllnLE8try0fLV8vPxJ7Go68RHWI3wDe9IVLRfU2zM7zVMxXksu5tUuI8RXmve/9ZOBZGNBXNDMv6sM7k11cMmtH1hmf6XlOZmfhYZ/Lr2Yxeo/KZlZvh5Jl7q5Kt3VzjOo29Jr159awxZa4zaGbQesHM8uu2M/yWz13yvJOnZiZS11lmC0uaimbmZX02MXNEZs4nZqZlRtfLnKWqmbslZj7L+kxi5rjM6HqZs6x7xm6MH7oeYx+/lbvGYV23PMhfxx5/0Jn8ejaj1yg1E+QHF7pe9jqyflHXLZdkRtdTz/NGZ7R4up66Djoq9N4TPbdM6lpb2Nm+y03d9w9wG2jp8tFZABWyks1J6eZ0BkANAr+rCTSP4gEOKB7ggOIBDige4IDiAQ4oHuCA4gEOKB7ggOIBDige4IDiAQ4oHuCA4gEOKB7ggOIBDige4IDiAQ4oHuCA4gEOKB7ggOIBDige4IDiAQ4oHuCA4gEOKB7ggOIBDige4IDiAQ4oHuBgvcX7Cy+22hQxbpxtAAAAAElFTkSuQmCC>

[image52]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAkCAYAAABBn/B2AAAAHklEQVR4XmNgYGD4j4TpDwyR8CgYlkADCdMfYCRvAIvxBwYPXESzAAAAAElFTkSuQmCC>

[image53]: <data:image/png;base64,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>

[image54]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAvCAYAAAArWAC1AAAAKUlEQVR4XmNgYGDgw4KHIVBBwvQH/5Ew/cHA2q6LBQ9DwIqE6Q8wohgAyIcKmv1nIEwAAAAASUVORK5CYII=>

[image55]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAFCAYAAAAdfF8qAAAAMklEQVR4XmNgGAUEwX90gUEMONAFKAHEeNwNXYAMoI4uQAYg5HEhdAF8YNTjeMCQ8TgAZN4FmrMHvyAAAAAASUVORK5CYII=>

[image56]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAECAYAAACQli8lAAAALklEQVR4XmNgoAHQUFbXQhejGQBa9h+Kw2ECGjTCMItWoLmBegBoOB8QyyOLAQBgSxaRysN0xAAAAABJRU5ErkJggg==>

[image57]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMUAAAAdCAYAAAAaVaiyAAACu0lEQVR4Xu2cMYsTURSFR7FZF21FLGTZDcZSEFxstZT9A3aCbGGjWNhpIZb+Ai0ULCxEBCtdQUWwc9WfYG1pJ7jxRDLrzcl7k5nMfZO8zfngNO/ed1ku50yyCaQo8uICNKjQzv/WCbjX6qfpEyIbvheTZmbNGopSQmQFGziktqEYShh6p9cfQYOIfnG/6I7nxbhxH0d0q7wQgHutFAoChj8bCEFMn/m+SM/XIq1xU87OjoDpV7lnCM6/UN8W94h0KBQdQSZf4TqDnqN05zz3iDRUhcKe1xVTVVsaYOg9Y+4TXK/CBoNrIg0KRWJg5r4x9m2u18Hc/8014U9VKDxIOTsLPJ70uHu37QxRn7qhsD0hxajTc6AxodjkWhPMnKpPAkWHcAhYIgAMvOb1hPd4xRFi7sDAb72MjBkPvGYJMTecQ3Hfa5aYDX571ERiBAz81MvImPHea5YQcwMGPu5l5HKOxywh5oox8zWuNcHMucQ10R38tmiaRACPJzzufmg7Q9TnWxE3Npt+mkJMqx94YORVE4w3XK+Duf+Ja8KfqlB4kHJ2NsDMV4yx97geA703zb2l3mGXxEJhz5rKEjtfOmDq3SYGt71GF7lP+KNQdAhM/Spg9NfQ5kj730VUSMFITCwUXqScnSUw9aGA0WO6PrqzQ+cKRkIUijkBY1+G/gSCMNSdQP9YMLgu/HhRjBv3YUTb5YUA3GulUDiiYHSHNW5M7/a7J+HemIQDSx2M/vqZH9BhPk/AqWLSwKy2oVg4sNstaIPPcwBhWLHB6C3o/xjY70k+aw2GDka6wbUEXIVeRnTP9DHcW+oZdMz0LRxmv8O/NytyeMUw+31SHvQdVA6telIvJdjJWmBfTVXuN8sfH7PB4FpbsJMjgX01Vblfn59PxaCP0Dk+Fz5gtz1oph8SEPXAfv99pFzyF5n3eY4gF/ReAAAAAElFTkSuQmCC>

[image58]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAaCAYAAACgoey0AAAA3ElEQVR4Xu2SMQ6CMBSGG6/BSsLgBcS1a0k8gKOb4WAmOnMCkHtwDRP0NdjAz0tF8ooMfsmXNND2ow1KfaYVdKcmYhdIEccx7ffACNKe81w0TERRRHtuMUZsXraX681KY2loT2gOo0uGB9GlwqMohk12+FqOftj+SBgNHX4PWOu6dgtw0ylyuHBRFCObpsH5YrgwB375XDn+4cXwhqvqLiKHN4xXNleOdYZD4g2XZSUihzeMVzZXjnWGQ/LzcJamKb4LRpIkFN1T2J7aGINzxNFau9P2OXUPQ3pUHU8pGJFKp0WaxwAAAABJRU5ErkJggg==>

[image59]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAaCAYAAACgoey0AAAAyUlEQVR4Xu3QTQrCMBCG4dwrV1B0mQtkJ+6FSMVt1l7Ajfh3o9whR4hMaEqcDtGGJLroCx+Uavu0ZexzruAs+zJ/QamMMeEBkrnr41kUhqSUcM8dxkIerQFDcE8MbuFkQFvCb2greIRieLFcTx5VDI9A2OV2rwqHA3Jx3eHo1L6bNKoBbl0Sxp8sd1Qz3KwkjJ88d1Qz3KwkbK0tMqokjD9Z7qj+E67Zz+GNEAL/Vi3OOaArgP1ba63xf4qnlBreNu7cn6y5E+t7AbDQlbXkbAGoAAAAAElFTkSuQmCC>

[image60]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAaCAYAAACgoey0AAABAElEQVR4Xt2RwQqCMBjH9xw9kKeCrr1VIHgQj16CIgKhBzB2U7BLpJ0Ejx58AmtTx/qcy+RTqR/8Qeb2/Tb+hHymRMyG9IQfwMIwjOYCWsqtaaKKGZZlsZknKGMsXil3+wMP+8aGzQROjpBOKX6TTiXmnerEy9X662RZJikrZLFSiiH2/YukrGjEzYcyRVGIA3BonwRBKCkrhNi27VbSNIX70RBiSJ7nrZsPja5juF5G0bU1YGi0HUOS5NEaMDTajqdGK47jBCUqOsX/0XH4Kx3f4xglKjrFs3U8m3hsZhdTz/Pgv9GglDLpjYn5qx3HgXvQcV1XvFbmWC+OmTOpeQJkipnStswEXQAAAABJRU5ErkJggg==>

[image61]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF4AAAAVCAYAAADPTVnBAAABAklEQVR4Xu2WPQ/BUBiFLdhsJjH4SEgsYmFiZBGLzSQx2Wwmq83EZhSLyY/yJyTCeWlEj/W9dzpP8ra9T9vlSdM2kwHNWqNuexEWdL6wKKeECAI6n1gUU0IEAZ2PLAopIYKAzgcW+ZQQQUDnHTsRAYTfstNTHwE03rDTez4CaLxmpz+bCKDxip3JEjvhCxov2ZmsshO+oPGCnckWO+ELGs/YmeywE76g8ZSdyR474Qsaj9mZHLATvqDxkJ3JETvhCxr3fxfvjyr2E0zue0K4grZZTDc5btumiHli7pgr3yB8QNs95pG0/vxBJosnXSuc+euMRQVz+7lGBACNz5g5exGRF1d1LUO1nW8vAAAAAElFTkSuQmCC>

[image62]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAECAYAAAAj3gc/AAAAMElEQVR4XmNgGMJAQ1l9C7qA8hDGR4H4PxBfQ/HUUARAT9yDeqYPXW7IAaAnspH5ACBzJRCJjG22AAAAAElFTkSuQmCC>

[image63]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEoAAAAJCAYAAACVHLqzAAAA/UlEQVR4Xu3VsUsCYRjHcTUw0MChRUEdFCHdiuCdBCcnBRdBXJoaBUEMBCGiIBCEQAiisX9BcHBwdInAv6KpqX+g79sl5L1PHMgp2t3wueN+z3Mc78PdewFVeQ74nBmBT/SpDwmEhKLXJfGBmL5ehnpQKcSFG7ymi3cc/87tTdoRTpARav/ZHG9C/s0IbPTQTnGGsFDfZxHMsEBaqK8wAgdZFFFCDkGhZ1cV8IJXTBEVev5kBGvSP4Rz1FBHA01lDXQb+57eX6voYIzJz/lCufQlGMGGHCKvrMW00FbWoq7QQx/XuMEd7jHAEA8Y4RFPuMUlysp6qw+U+TzXfQHaFAKi23oBSAAAAABJRU5ErkJggg==>

[image64]: <data:image/png;base64,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>

[image65]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPAAAAAfCAYAAADZR0DFAAAFZElEQVR4Xu2cR6hkRRSGDyoqBkwYQJwZdGHEAOpKMKxUVBTBLA4IyowoYlwognGjYELBACpmxbRQF8roQhxRUVTEiFkMGDDnUB9V573Tp+t2uK9fv+7H+eDQVX+97um+Xf+tqlPVIxIEQRB0sIkXxsCtXgiCYHh2SrHMi0Pyd4p3vDgA73khCILB2TLFEi8OyQamjJHXNXXlgxT/pvjSNyS29kIQBIPxixfmyMMp7nTaj+XxgBRP24bCr14IgmAwzvBCC9aWPPIemuK/FB+btqUpXjD1Gsu9EARBf3ZLsZYXW4BplcdSfGbqX6XY3NRrrJFiDy8GQdCbG73QgstT/GPqqyVPoxVr7l7c5IUgCHrzohda8J10GpZE1WalvFGKr01bL172QhAEvXnTCy14JMWDpfxointN2+8pjpQ8RVb+MmVLmy2oYDIgz9GL2q7EgsGUsBaW043OelA78CtGp8MuK/pC8YcXWvKu5M90gtHOTrFpiv0kZ5lpbzIvkAQLpg+WUBt7MXFyirNK+ZYU65u2ieDuFNd6UbIpnjD1E1PsVcrLpdPsq6Rz/TgIPGdUjMrAoyAMPH2sk+ILL0peRvlRF62L+8vj8zJ+h9cMfL50j8YW9kF9OyZ6w2kKn4815kWl/qzkC2H3WQ+TPLLdZ7SjUqyX4qkUP0vzMcm3vFDQ67q9jCbRNQifeyGYeNhh2NFpNfMCOQ62GWdg5HqolH9Lsbdpa8uGKT6tRI2agRlFMFwTNQOfV9GAAxaYEJimwPHSOWLfIXl6AufK7N1Qp+jAPi917paemoHtdeVGwJc0DsLA04fvt29L7nec7DvOtR0uzhs8eYtSXogvv2Zg3tNHTrPUDHxwRQOMZA9EgDcwz9P19Valrro93shz7jF1pWZge113kdl1zHyzEN9h0B76ne+31P+U3G9uL3WdGW8nZhq9ZmlURnUckNdlmPdRo2ZgjMKU1UImlhkC1Ax8QUUD3gsfmLbnilYzcA1v4G+lvmXkDeyv6+umPN+EgaeLJgOTuFToq4+X8s5iDMy0UU/70KkXAqaZ/vDBvtL9od5P8WEpHyjd7dRPcpqHC7GD1A28j6kr3sBcOE2kWbyB/XXl84yLMPD0Qb/SZR7Q7w4xdfqq5m/oW69qAwtiTdxco6LkF2C04YWfLNqVkuflLKy5C2gH1axn247Da9YOH7wknRk33o/ut14iswZmakr5nFKH74sGvAZnjLUM20puP0XyMUjWxtRZ+z4js2sMtAtL+cwUP5Wyxxu46brCqhRHOM1yveT3eZVvGBD7PfDF32zqwWRyW4obTP1Yme2ru5oy4DdyTDOcaisFpqvA8M4JIEZJXuhoyQcOgBelnc7OOWCfRRsVK1Ps78U+YFjer8L6+BhTB37+t6fTTpPOqQsGXio5gdVx0Ry1baTadQX9RVET+mXx2+JBT19Z7DaS3sSCyceaVGEPmMFSoc9yVqAvutZUA9ee9EmKyyQnfe5ybYsFDMBF60fNwE005QJqtDGgNbCd/geTj0+2eppmgF1oJ8DATEdZH15ctOvKI2tBprlQu3ssBjCQPRXVxDAG7gfbePy7BNd1ieRZATdVthCgl7HjIEcQDMkPXpgDPrm2u3RnKnsZ2GfvgyDogx4QmSuYV/ejAaOynmctpHkJtqgemPmLbq72QhAEvSGrrZnuuWBHVt2/BvIQ5BvgtfJYgx0Cjm0GQTAkl3qhBdbAJA5XGH2bUiZT3oRueQVBMCTsU8+VKyT/wOIb6czsMxJj4n5rbRKOQRC0gOnrQV4cM+P+JVkQLCoYQW0SapzE6BsEI0J/hTQO+P+yOGkWBEEQBEEQzDv/Az+wRLGOS76KAAAAAElFTkSuQmCC>

[image66]: <data:image/png;base64,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>

[image67]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAADCAYAAACXr/8TAAAAJ0lEQVR4XmOQZ1jnC8T/qYxvM4AAkLEYiyTZGGwoDKBLUoA7QeYBAFR9a/UETW/AAAAAAElFTkSuQmCC>

[image68]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAPCAYAAADUFP50AAAAi0lEQVR4XpWNMQ6DQAwEr+Ed0PKK8KA0VPkV/6BIBzVFKBE1VcCRAC0Tg7iVViedZ+wQJGmoHtb5Rhv1fiIBt5QaAifNVCq40d6X02SXVojSoH86U6nlkMLamiKBj3W0Tlz4Fw/Asqfye2yQA9R+yR+ioIrk3DjX3mTcGNhFX9siYsnZZUzoo6/FZAHberUWhWlzwAAAAABJRU5ErkJggg==>

[image69]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAATCAYAAACUef2IAAAAxklEQVR4Xr2SMRLBQBhGV61SaqSIGQdSMo7gChpnMSoHcQIKCoXGDOMAGb6EZDbPbiay4c28Ipv/f9kiJjKbq3y0rUnhYahZ9B3u82WIRZhg8ML3jVDozK/zRj7ZKoHhiRwz4HHNVgl7mM8O7/Igt+x8gOgJoYGcOxyy40SDR9lBtIvnQu5XguWb4yw35m4lWE4853t7pzaI9OTUPuN8bbS8Q9x2wfmvcATDbpujyIxROeJcI6LX79febW1+Ek1RdCWXPP87T28mQGlHFMXsAAAAAElFTkSuQmCC>

[image70]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAANkAAAA0CAYAAAAHZJ0qAAAGC0lEQVR4Xu3dBYwcdRQG8OJOseLwDoJr0eIcBIfg7gR3d2lxaXAIFAsORYoTnOAQXIPT4O7u37u325t5///63O3O7veSX2/v7pu99prvZnf2P3MDZMCoAdTvJoYuGAwrw0awMxwKx8BQGAbHwnFwPJwAJ8JJcDKcAqfCaTAcTocz4MzC7aNhL9gK1oKlYTYJ/y6UsWB8gDIxCNaF/WB/OAAOhIPgYDgEDoPD4Qg4Eo4SK0YWJTsLzoZz4Fw4D86HC+BCGAEXwcVwSeHtDjBQwn8L1SgYH6CKFoEdYRfYFXaD3WEP2FNs77E37Cv5KdmlcBlcDlfAlXAVXA3XwLVwndjX0z2v/55QQjA+QCkCW8L2Yj/ptVw7iT20q7dkuu3G0A0Li32NySX82lmYEGaA+WF52FSsoPWW7HoYCTfATXCz2L/Zf92OFowPdLglxZ7DbA3bwLawndResjVhSgnvv5XNLbY3rbVko+AWuBVuE/te+PvuKMH4QAfREmgZNoPNYQuxvVY1JdOsPmz099muphXbM1dTstvhDrgT7hJ7uOrvr60F4wNtbHHYADYUO5qnD9k2kfIlWxHGLWxPcWtL+ZLdDffAvXAf3C+2p/T30zaC8YE2s6jYUb71YH0pXzJ9O4fbnmqn3+NKJXsAHoSHxPaE/j5yLRgfyLkJYFWx14X0J+w6Urpk+v6kbnvK3gJSvmQPwyPwqNgRVb997gTjAznUDavB6rCG2POsWMkWjGxLzaE/7GIlewwehyfgSbEfiH7blheMD+TEnGKv16witucqVbJ5I9tSa9H/t1Ilewqehhsj27WsYHyghU0kdiCiG1aS0iWbIrIt5YO+GB8r2TPwbOG236blBOMDLUhfqNUXUleQ0iUbO7Id5ZseqfQlew6ehxfE1n/6bVpCMD7QImaCZWBZWE7CkuliV78NtS99ecCX7EV4SexFc59vqmB8oMl0lfhSYiWKlWySyDbUOXT9pi/Zy/CK2F7P55siGB9oEn24twQMkXjJfJ46mx519CV7FV4TWxrm8/0qGB9ogsXEVmP4knXSsiWqT5eEJXsd3ih8zuf7RTA+0E90ZYWWSFdk+JL5LFE1fMnehLfEnt/7bJ8Kxgf62Kxip3cMlrBkPEJIjZpMwpK9De9Esn0mGB/oI2OJrbhYSMKSafF8nqgRurjbl+xdsXP7fDZzwfhAxsaH+cROGvQl81mirOk5f8mSvQfviy1c8NnMBOMDGZpHbFmTL5nPEfU13aslS/ZB4bbPZSIYH8jAVGJn2fqS6Qp5nyXqL7qmNVmy0fBhJNewYHygQbpwdy4JS+ZzRM3iS/aR2E7A5+oWjA/USVdq6GF5XzKfI2oVo6W3ZB/DJ5FMXYLxgRrpMict2OySLhkfGlIe+JJ9GsnULBgfqIEeeu+SdMn0ts8RtTJdLZIs2WdiCyN8rmrB+ECVtGB6vcAu6S0Zz+OivBom6ZJ9Lg0sRg/GByqYGWaRdMn0aKLPEeVRsmRfSp3P04LxgTJ0DZgvmc8Q5Z0uOC6W7KvCW58pKxgfiNAzUGeUdMn0rc8RtQs9b61Ysq/hm0impGB8wJlO7FrqyZLxyCF1Al+ybyOZqGB8oEAvWqOXZp5e0iXzOaJ2ppeuS5bsO6liJxOMD4hdI36QpEvGi4BSp9LLXiRL9n0kkxKMC2iZppF0yXyGqNPo4vZkyX6IZMYIJvFJPRQ/tfSWjK97EaUlS/Zj5PM9gsEHxxF7iJgs2Xh+QyLqOXs/WbLoQ8dgxPZYyZIFGxHRGLo2t1iyn8ReV/OZFP0jWTJeZ4Oosn2kt2Q/i/2+bp9RPQ8p9YaWbGAkQESl6XUfiyX7RcITQP+Cv/W2vlPxuD8RRelzsmLJfhO76pp+XH/Nb6pkRFS/ZMl+L3zsT2HJiDKVLNl/wpIRZW64WLmKWDKiBujzLv0VTsmji7on01L9C3+ILxmHw6l+CkXTxfK+ZMWHiywZh9PIFEqWNJIl43AynEjJir5gyTicDCZSLs+XbIi/Dw6HU2YipYr5NVGyEf4+OBxOmYkUqhzdq/3j74PD4ZSZSJEqGfo/1zMJBlWnz2IAAAAASUVORK5CYII=>

[image71]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAN8AAABRCAYAAABSQ3OEAAALY0lEQVR4Xu2dCcwkRRXH3yLenEYRlbio4Ek8EGU9WSQq3hpMTIziqgmoKPGIWXUjn0ENaAIeaCQmBkxAJagI4hnQLFHwgAhGDRJ18SB4RhRvdsD6f9Vvur7X9Wp6Zvqoqqlf8rL7vX/NdL3qetPd1dXVROmxv3QUCh1yN2NXG7vD2CXGnrNRXi1OIdsQ0gqFZbm3sQup2bek7ccfyJn3UzNw115XleO/C4W2PNTYZdTsU2wTY2+blq7Jrq9tMnYGNRvAtbOM7ckfEOxDGTZKoTOONHYNNfsU2z+NvWZaejZJ9rXHGfs+NYNnwy/RIdPS85NkoxQ6Y42afco1XLJ0BX8nDhzRscXYddRsALZLjR00Ld0Nd6H6+xNlst2YUn/Nv3LcydiHqdmnXDvH2D35Az0RRV/baux6ajYA243Gnl2V7ZvDyW7zP1JIg5J8CldRs1+xnWlsj7rooHAdBudgajYEknBrXWQUTidbl+9JIX5K8nl4A9X9691CG5t70AgJiHNdbPDlUogEPhK/WQpxU5LPw+Cd28N3pcPhqWTr9w8p9MXvafwGmbV9DBejzEOkEC+cfJqtJIj7E9I5IDeQrcMbpeDwMbJldkqhD7Ch26VzYFCHWR2Sy0Q5KtWkHPkE76HZ+3gI+If8QVJw+BXZMnz/uRcw5QYbua8URmCeBEyAknyCmPZdmx9yHJBQ5mDh74yYGgS0qQ+XuasU4qIknwOPKzxCCiOCUXTUCaPqGnwv+wNS6AJ88UXSOTLzJGDElORz+APp+wvzMMeC+xHuK2v8m2yZI6SwLFqDgC9Jx0C0HfJtU6YQB9hP2rjC2Puxzfa5TGdnWyeTvtHTSNeG4Glkt3+rFBwwO6JNw43EZEs1svlwqazgkQ/xauMK0K6QzoFp04/alGkNvuiT0lnR6YYW5GyydbhcCg6PJFvmNimMzzT5PO3o82XLXqT3peNI14YEE/3b9HnouDW3NKENQXundI7Ar8nW5dVScNhBtszeUhgXN/kmGLp2tVDb58Z/SZ8i2KbDD8VhZOvyPyk4YOobynxOCvPwPtKDPpd0bQx4Bz1ACg7XUlx1pjr51v+PBMRpMmuR1bVXECuu431Au1g6R2SNbJ1+JAWHP5Mt80IptIU7tI+QNhZt6tSmzIBsSL6dGxNuZZLvfqTvk9eSro3Jj8nWK3Tmx33tXlKYBR868QSwZF+yGv6NDZwOoG64ztMYKQEn+xm7wDnNPGpj8q2Xgf/v9f9XgtD+CGljwz8Mu6Tg8AuyZU6SQgg+bPrAwIWmxQDvMOcUrkHPO7WRaNLML+fksc3kW/8s9E1Nf7Ygzo9KZwW0mBcx+hbZOn5cCg48Ve3BUtBAYXzIB7S/SmdkoI6zOm+bMi2YmWi7jSmTdL3Jd2H92ewJjSv8kHQtJvB0A+r5FCk4cF8LTVVbh29g+2YVYKIpNG19lZjggDWwKtWsMhI03jeo/pyx224XiTbHKYYv+db91fdlT6j9Q1pscF3vLgWHVvHwfDYfrb4gEtosM/E8svqfpED2R+hTVH+HNLTTkjPacXN98lXpNb4D/P7sQDtqD8xC2yKdETOrr4GZZSBihScf0H4pnRHzeLJ1xtw7DdyPQZmbq381ey9/YDgme1ZHwSdJJQMweUPriDxQkRqzkgtHRlU/kKyIx4gkT6bAByPmg2TrHXoK300yDCi9aaM8JtNTWu0+WKqEOmpIixlclrxMOgWI6wTpBPx8ko9UGwQcKh0CxIUjX6RkeQ2IeE6UTqpvc3nmuyYPng5S9yMEbdgUWkRHhM7glbIjZ3oExNlJ6uQyrjAviMt7SXcq6UHzmvY5ktDOzuYIqHZCslpoAaNUwRMbiM13SRfshCEtdRCXNuIWIcknYGhc4UWUbz8LXdKtC2vSWQHtpdKZAaERt4hJOgFDnTD3H/k/SifAUm1a0N8hXUudRHf2ZO8qAXGpkBpob+15N2hfkc4MeBjZ2LwrbYc6YUhLHcTlG3FLgMnRVQLihTSp8Ciybe6bZrWN8u5namwQsEy3D2iheWupgvVn1AZJg+npZ2hxn5gIdcKQljqIC4vxNvgChYMOaSmTy85OKQ7U82fSWQHtPOnMABy41P0D4V/S6fAO6ciA0IhbinAC4sWhsXI2BTphxqg/jgeQFUKzsXMkNOKWKupOjoTY69cXiNn7Ap8cO2EbELM24pYyMXRwDKZg9Waui7RV4jMUiJnf+Hm+FDImNOKWA0N38uOpmWCufYjsEHvbhY5zYma8fyFb4FgpZMrMBkmcB5KNT1v9eRkwG5/nZfrs9Lqol6PIlqvWqckexPoK6ZRw48W4KFLXIE5txC0XXk82TjwXtyiYRL+bmgnG9mUKL9WowbOKvimFzOAVD1rBjZozWyn/GBmemXSGFDy8nerrf599nvzLiizKb8h+76ukkBFz59PcH0iM3ONjsD4NwC0kxPsER3su1Z3fZ7jXNsQtC97e/aWQCYht7pXXcu6giGundGaIO4otk8s1rLo85m2mXPvaNbREXLk2So4x+UCc2nvNsRxITMS+DuwiIJ7fSuc85JaAl1Je8WicQuE4oT1GOkcmt762dCy8LPx2KSQKYsnxmURJqCPvIl0bG9Srj9sjQ4PlDjtpY1ww4ovwPFLqdNIgCYA4sRK0j1Bijg0e7ULdUlqa0gdiuE46F4V3mPdBwET4AcXb6brkI6THiRk90EIvkRkbvH0WdcTMq1TR2n9hYv7FbEPq9W9LKE6+vRA7WFgJ9UxptWqGV0DvnNCOHYvN1G7ZAdT7SOnMEMT5FumsiHH/aXBdx3zkC2+54rnPbHiDboje2hgN0duXtwAjdFfSxsZoUx+8bnlWmRzAzXEtTj7lnLV4cEy02bddgul0fNvDZ2fS7EsvlPuadHZJ342C0dXQVCcscjsP+ExnF8ARgzi1d4X3vc/6guu9vxQW5CUUficHBnvuMy09H3ixTe9tzMt5d7Gh06jZAK59lpZr+GdSN/WMHX7dmTYtDNrV0pkIi/a1bVS/O89nmIXS5aXIovWcG7wuGhvCrPc24H1+vDyhZtC7XghosAYZmdAskWeQrqXCPPvRd+p4OfV/uwzbeb509gVmwmODP5WCgtsYOOINAbZ1mXTmhultu/fRpzPN03Fjpm0cmDiOpdmH4BCyj0a1rVun8Kt7tZu6Q4Inpc+hjUmO5QGzZrKJNhu7w9idpVaBdrhKOhOEF70avJNXHEF1f/fZKGD4FRsfcr4gpr7x+hiavWtaOmOqxPPufHMofBaN2DF6AE+FI57fSaFj0G78gk6fYdn346alR4Yrpf36LssOajYAG0ZG31oXXS2q5LtR+kEoMRPm62T3O17Z3QXy1FHaz40dMy0dKVzZZcHrl2UDsOFi+qS66GpjEuuJoeSqkg+dNTf+RrY/HC2FGeCh4tCpIx4+PnxaOjE4iL2kIMArmm+hZvBsuJn+6GnpgpfQkQ1Jp2mZ8GmyfQX311zwZDy/CkCztWnpzOAAGVwL8nw9n32b+h8GzpIq+byn3KHEzIibqNmfXMOtsJOnpVcE2QhsmHqz2SlXWIJQclXJtwrLQf6E6v71SqGtLBeQvedS6AGTWF/Uks/4r9W0QqGwJNWRDdfODVbklLNQGIdQclXJl9JLMwuFNDCJdaWWfMZ/k6YVCoUlCZ1WhrRCodATJuleXBKvUBiBctQrFHrEJNcexm7lRDN2mNC7fEC0UCgAk1jbq4S7wthBxtaqvzEnsVAo9EWVaPwGIunHG3cLhULXmOTagSSTfmD8N2taoVBYEpNct2gJZvwv0LRCobAk1amlN8GM/xhNKxQKS2KS61QtwYx/l6YVCoUOqI5+jaU6Kv/Tpb9QKHSESbDzqkTbVv19YOh0tFAodIibcJUdL8sU4uf/BBZStCKlyJgAAAAASUVORK5CYII=>

[image72]: <data:image/png;base64,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>

[image73]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAABMCAYAAADUQBHfAAAAw0lEQVR4Xu3cMU4CYRCG4bGm8gQewBtAYUtl7Tk8BiUV4QaeyZZDQCx0dUN2mD92dPM8ydssU33ZUG7E4jD1nWotD5F7zkdd7GMcovWbUgeobZfTHuoAtbfltIc6QK2lOsK1Sz7q5CnGMc43F02tpl7rQwAAAAAAAAAAAAAAAAAAAOAeXqZOU8f6Q0f1ww+/7W4uGqlD5B7TXQsPMY5Qa+U9xgFqrWxiHKDWTh0gt053bfz3P/KZjzr6iHmIr5g/NfTnBy+EbMc71+KKAAAAAElFTkSuQmCC>

[image74]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAARUlEQVR4XmNgQAApIH4LxK+BWAxJHAz+48BgkI1FAobjQArQBdExhgA6ZijFIgjDGSAFIIAuAdeNDHSB+A8Q/wJiDZggADz9LdJOSsmdAAAAAElFTkSuQmCC>

[image75]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAMCAYAAAC5tzfZAAAAaUlEQVR4XmNgwAH+MjLEoosRBEBN/9HF8AKghvcgTUB8FV0OJ4BqAGN0OTAASjgCcRUSrkbWhCYHwsYwjciK8GJ0G2egK0DDjSgaYAAosQKLYhCuQFcLB1gUwzG6WjhAU6hHkiZCYmQBAAcVehrSwYJmAAAAAElFTkSuQmCC>

[image76]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAUCAYAAAA+wTUXAAABI0lEQVR4XmNgYGAoBOImhhEGKoD4PxKeiSo9sgByQExDkxv2IIUB4nFxKH8FELMjpEcG+MaAmgrqUKWHP6gB4oVQNigAdiHJjRgAygIvGSABsBMq5oOQxgtY0AWGMkDOCueAmBVJjhlNHoaHDdAB4h9A7AzlL2bA9CwM34OqGXagkwHTs8MytpHBUwZMj6JjT7jqYQJWMmB6EhmfhdKGUPVHGIZJYYfuUWSMT81PIOZFUoMNvGVA1fMcVXrgALpnYDgOWRESmMsAie1TDKjqNyMrQpNDx4MCsAHxBwaIg66hyREDohkgev9BaUJ4WPYo0T2JDQ87AGoLoHsSGb9GKB0+QIkB06PIWBShdHiBLgZMz4JwObKi4QhABSSyh0HtfDAAANvilqTaJbDVAAAAAElFTkSuQmCC>

[image77]: <data:image/png;base64,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>

[image78]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAMCAYAAAC5tzfZAAAAdUlEQVR4XmNQZ9j6n0R8hAEEsEjgxGANyACnBC6AZJobuuk4DQMK/oFK1kNpBXQ1OAFUw18gXolkyxt0dXAAlKyBKuJBshnTWegAyXRWKD0XXQ0GACoSRNJI2BYYACr+AdW0Hl0OJwAq5iPJFhgAagpAF4MBADovc9cxbIyuAAAAAElFTkSuQmCC>

[image79]: <data:image/png;base64,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>

[image80]: <data:image/png;base64,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>

[image81]: <data:image/png;base64,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>

[image82]: <data:image/png;base64,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>

[image83]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAHCAYAAAA1WQxeAAAARElEQVR4XmOQZVz/DIj/48CfGGAAKiAGxF4wjCy5EIifQxXdAuJjQBwAVwBVtApkJMx4FEkYwCsJAkDJ+UDcjC6OFwAAH2wqlshALHMAAAAASUVORK5CYII=>

[image84]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAHCAYAAAA1WQxeAAAARUlEQVR4XmOQZVx/EYj/48DHGGAAKiAPxKowjCxZCMT5aLrV4Qqgiv4C8VKYAhRJqAIfJN086PJgAJS4jFU3DAAlWdDFADNNLPX8Ert2AAAAAElFTkSuQmCC>

[image85]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAHCAYAAAA1WQxeAAAARklEQVR4XmOQZVwvC8T/ceDvDDAAFXAD4qkwDJeEKlgJVbQCqsAeXcFTIL4GMx5FEgZgkkBsjC4HBkCJDTh1wwBQATMyHwBlUSs/hhKuIwAAAABJRU5ErkJggg==>

[image86]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAHCAYAAAA1WQxeAAAASUlEQVR4XmOQZVzvAMT/cWEGGIAKyAGxLwwjS7IB8QWoog9AfBeIS+AKoIqOAvFHDKORFDDCJIE4Cl0eDIASZ7DqRgZABQLIfAA3wC1CVrNYcgAAAABJRU5ErkJggg==>

[image87]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAHCAYAAAA1WQxeAAAAQklEQVR4XmOQZVyfB8T/cWEGGIAKsAAxGwwjS4IE3qDp3gBXAFU0G4j/YRiNpIAdSXcjujwYACVeYtWNDIAKNJD5AA/mLU/9QL67AAAAAElFTkSuQmCC>

[image88]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAHCAYAAAA1WQxeAAAATUlEQVR4XmOQZVy/DIj/48IMIABk2EMF/KFsGOYDK4AqmgFV1APElUCsC5eEKngHxCdQjEZTIAuV/AnEG9HlwQAosQKrbmQAVMCIzAcAz9gttrvZwNkAAAAASUVORK5CYII=>

[image89]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAMCAYAAAC9QufkAAAAeElEQVR4XmOQZVzfDcT/ycA/GEAAiwRBDNaICwAVWKBrQMK86OpRABYNMPwJXS0KACo4B1OMLkcQIGtEs1ULXS0KACp4BFX4DohXIGl8gK4WK0DSwIzsCqIAUHEuVNNzKN2JrgYvQLKdeFuRAdkaQQCoeQO6GDIAACPYfqMG8XewAAAAAElFTkSuQmCC>

[image90]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAMCAYAAAC9QufkAAAAgklEQVR4XmOQZVx/Aoj/k4MZQABdkEjMCNaMDQAlBYF4IQ7Mi64eBWCxCY7R1aIAoIKpSIpdgdgQiOWAmAtdLQoAKhBCtgHNViZ09SgAqOAPVOExNI2/0NViBdhsRleDEwAVy0A1vYPSRuhq8AJkJ6PLEQWgmpnRxYkCQI3h6GLIAAB1v4Egr+a2QQAAAABJRU5ErkJggg==>

[image91]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAMCAYAAAC9QufkAAAAfUlEQVR4XmOQZVz/Aoj/k4MZQABdkEjMCdaMDQAlmYE4GgfmRVePArDYBMfoalEAUEE2kuJuIC4H4hQgDkBXiwJkIc6F24BmK9HOXQbEX5E1o6vFCrDZjK4GJwAqtoFqegqlw9HV4AUkOxcZADUxQjXzo8sRBYAa69HFkAEADgSBnOOP23IAAAAASUVORK5CYII=>

[image92]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAOCAYAAAAmL5yKAAAAdUlEQVR4XmOQZVwfDMT/ycRNDCCARYIoDNZMMwC0gRHdRiT8F109BsCiiXinAxVFE60YG0CyrZgc25uQNAgA8Ukkvh66eqwAScN5IP5KtO0wAFQsgmQIDDOhq8MLoJoeQOlf6PJEAZKdjg6Ami8C8X50caoCALTljKXmfVM4AAAAAElFTkSuQmCC>

[image93]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMCAYAAABfnvydAAAAPklEQVR4XmNgGApARV5hGroYDADl5sEY/6G4GYiTYXx01TBFMCyIriAaSXIziiQMYDUaGQAlM4H4ELo4XgAAeZEYeJrwtEEAAAAASUVORK5CYII=>

[image94]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMCAYAAABfnvydAAAAR0lEQVR4XmNgQAMq8gpK6GIoYCAUAAUa0fhwBUD2ZBjjPxRPBeJMGB+uC00RDOuhK9BBkryPIgkDWI1GBkBJcyB+jS6OFwAAEOEbnwS5AyoAAAAASUVORK5CYII=>

[image95]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMCAYAAABfnvydAAAAOElEQVR4XmNgGApARV6hC10MBoBynTDGeyD+j47RVeOWhCrIR1KwF10eDHDqhgGgpBAQh6CL4wUAwVIYRI1FeroAAAAASUVORK5CYII=>

[image96]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAOCAYAAAAmL5yKAAAAdklEQVR4XmOQZVw/DYj/k4k3MYAAFgmiMFgzDAAFzEjEpigGYAPItqFjdLUYAF0DqZrViFaMDSDZtokc2/cgaWBHMyAFXT1WgKRhKhBfJNp2GAAqtkOzGYRF0dXhBVBNL0i2HQaAmhjJ1gwDQM1fgPgVujhVAQDbZZshN7boVQAAAABJRU5ErkJggg==>

[image97]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAOCAYAAAAmL5yKAAAAdklEQVR4XmOQZVw/B4j/k4m3MYAAFgmiMFgzDAAF7EjEtigGYAPItqFjdLUYAF0DqZqViFaMDSDZtp0c2w8gaWBFMyATXT1WgKRhAhCfJdp2GAAqdkazGYQl0NXhBVBNL0m2HQaAmhjJ1gwDQM3fgPgtujhVAQA31JugQwX38wAAAABJRU5ErkJggg==>