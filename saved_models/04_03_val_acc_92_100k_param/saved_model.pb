Òå
Õ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*	2.3.0-rc02v1.12.1-34938-g99fea8da0d8Þ

conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_26/kernel
~
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*'
_output_shapes
:*
dtype0
u
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_26/bias
n
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes	
:*
dtype0

batch_normalization_38/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_38/gamma

0batch_normalization_38/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_38/gamma*
_output_shapes	
:*
dtype0

batch_normalization_38/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_38/beta

/batch_normalization_38/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_38/beta*
_output_shapes	
:*
dtype0

"batch_normalization_38/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_38/moving_mean

6batch_normalization_38/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_38/moving_mean*
_output_shapes	
:*
dtype0
¥
&batch_normalization_38/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_38/moving_variance

:batch_normalization_38/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_38/moving_variance*
_output_shapes	
:*
dtype0

conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_27/kernel
~
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*'
_output_shapes
:@*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:@*
dtype0

batch_normalization_39/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_39/gamma

0batch_normalization_39/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_39/gamma*
_output_shapes
:@*
dtype0

batch_normalization_39/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_39/beta

/batch_normalization_39/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_39/beta*
_output_shapes
:@*
dtype0

"batch_normalization_39/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_39/moving_mean

6batch_normalization_39/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_39/moving_mean*
_output_shapes
:@*
dtype0
¤
&batch_normalization_39/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_39/moving_variance

:batch_normalization_39/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_39/moving_variance*
_output_shapes
:@*
dtype0
z
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@d* 
shared_namedense_38/kernel
s
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes

:@d*
dtype0
r
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
:d*
dtype0
{
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ð* 
shared_namedense_39/kernel
t
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes
:	Ð*
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
:*
dtype0

batch_normalization_40/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_40/gamma

0batch_normalization_40/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_40/gamma*
_output_shapes
:*
dtype0

batch_normalization_40/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_40/beta

/batch_normalization_40/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_40/beta*
_output_shapes
:*
dtype0

"batch_normalization_40/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_40/moving_mean

6batch_normalization_40/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_40/moving_mean*
_output_shapes
:*
dtype0
¤
&batch_normalization_40/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_40/moving_variance

:batch_normalization_40/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_40/moving_variance*
_output_shapes
:*
dtype0
z
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_40/kernel
s
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes

:*
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

SGD/conv2d_26/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/conv2d_26/kernel/momentum

1SGD/conv2d_26/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_26/kernel/momentum*'
_output_shapes
:*
dtype0

SGD/conv2d_26/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameSGD/conv2d_26/bias/momentum

/SGD/conv2d_26/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_26/bias/momentum*
_output_shapes	
:*
dtype0
«
)SGD/batch_normalization_38/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_38/gamma/momentum
¤
=SGD/batch_normalization_38/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_38/gamma/momentum*
_output_shapes	
:*
dtype0
©
(SGD/batch_normalization_38/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_38/beta/momentum
¢
<SGD/batch_normalization_38/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_38/beta/momentum*
_output_shapes	
:*
dtype0

SGD/conv2d_27/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameSGD/conv2d_27/kernel/momentum

1SGD/conv2d_27/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_27/kernel/momentum*'
_output_shapes
:@*
dtype0

SGD/conv2d_27/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameSGD/conv2d_27/bias/momentum

/SGD/conv2d_27/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv2d_27/bias/momentum*
_output_shapes
:@*
dtype0
ª
)SGD/batch_normalization_39/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)SGD/batch_normalization_39/gamma/momentum
£
=SGD/batch_normalization_39/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_39/gamma/momentum*
_output_shapes
:@*
dtype0
¨
(SGD/batch_normalization_39/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(SGD/batch_normalization_39/beta/momentum
¡
<SGD/batch_normalization_39/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_39/beta/momentum*
_output_shapes
:@*
dtype0

SGD/dense_38/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@d*-
shared_nameSGD/dense_38/kernel/momentum

0SGD/dense_38/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_38/kernel/momentum*
_output_shapes

:@d*
dtype0

SGD/dense_38/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*+
shared_nameSGD/dense_38/bias/momentum

.SGD/dense_38/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_38/bias/momentum*
_output_shapes
:d*
dtype0

SGD/dense_39/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ð*-
shared_nameSGD/dense_39/kernel/momentum

0SGD/dense_39/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_39/kernel/momentum*
_output_shapes
:	Ð*
dtype0

SGD/dense_39/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_39/bias/momentum

.SGD/dense_39/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_39/bias/momentum*
_output_shapes
:*
dtype0
ª
)SGD/batch_normalization_40/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)SGD/batch_normalization_40/gamma/momentum
£
=SGD/batch_normalization_40/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_40/gamma/momentum*
_output_shapes
:*
dtype0
¨
(SGD/batch_normalization_40/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/batch_normalization_40/beta/momentum
¡
<SGD/batch_normalization_40/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_40/beta/momentum*
_output_shapes
:*
dtype0

SGD/dense_40/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameSGD/dense_40/kernel/momentum

0SGD/dense_40/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_40/kernel/momentum*
_output_shapes

:*
dtype0

SGD/dense_40/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_40/bias/momentum

.SGD/dense_40/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_40/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
X
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*×W
valueÍWBÊW BÃW
û
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api

axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%trainable_variables
&	variables
'	keras_api
R
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api

2axis
	3gamma
4beta
5moving_mean
6moving_variance
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
h

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api

Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
R
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
h

\kernel
]bias
^regularization_losses
_trainable_variables
`	variables
a	keras_api
Æ
biter
	cdecay
dlearning_rate
emomentummomentum·momentum¸ momentum¹!momentumº,momentum»-momentum¼3momentum½4momentum¾?momentum¿@momentumÀImomentumÁJmomentumÂPmomentumÃQmomentumÄ\momentumÅ]momentumÆ
 
v
0
1
 2
!3
,4
-5
36
47
?8
@9
I10
J11
P12
Q13
\14
]15
¦
0
1
 2
!3
"4
#5
,6
-7
38
49
510
611
?12
@13
I14
J15
P16
Q17
R18
S19
\20
]21
­
fmetrics
regularization_losses
glayer_regularization_losses
hnon_trainable_variables
trainable_variables
ilayer_metrics
	variables

jlayers
 
 
 
 
­
kmetrics
regularization_losses
lnon_trainable_variables
trainable_variables

mlayers
nlayer_metrics
	variables
olayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_26/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
pmetrics
regularization_losses
qnon_trainable_variables
trainable_variables

rlayers
slayer_metrics
	variables
tlayer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_38/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_38/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_38/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_38/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
"2
#3
­
umetrics
$regularization_losses
vnon_trainable_variables
%trainable_variables

wlayers
xlayer_metrics
&	variables
ylayer_regularization_losses
 
 
 
­
zmetrics
(regularization_losses
{non_trainable_variables
)trainable_variables

|layers
}layer_metrics
*	variables
~layer_regularization_losses
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
±
metrics
.regularization_losses
non_trainable_variables
/trainable_variables
layers
layer_metrics
0	variables
 layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_39/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_39/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_39/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_39/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
52
63
²
metrics
7regularization_losses
non_trainable_variables
8trainable_variables
layers
layer_metrics
9	variables
 layer_regularization_losses
 
 
 
²
metrics
;regularization_losses
non_trainable_variables
<trainable_variables
layers
layer_metrics
=	variables
 layer_regularization_losses
[Y
VARIABLE_VALUEdense_38/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_38/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
²
metrics
Aregularization_losses
non_trainable_variables
Btrainable_variables
layers
layer_metrics
C	variables
 layer_regularization_losses
 
 
 
²
metrics
Eregularization_losses
non_trainable_variables
Ftrainable_variables
layers
layer_metrics
G	variables
 layer_regularization_losses
[Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

I0
J1
²
metrics
Kregularization_losses
non_trainable_variables
Ltrainable_variables
layers
layer_metrics
M	variables
 layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_40/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_40/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_40/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_40/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
R2
S3
²
metrics
Tregularization_losses
non_trainable_variables
Utrainable_variables
layers
 layer_metrics
V	variables
 ¡layer_regularization_losses
 
 
 
²
¢metrics
Xregularization_losses
£non_trainable_variables
Ytrainable_variables
¤layers
¥layer_metrics
Z	variables
 ¦layer_regularization_losses
[Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

\0
]1

\0
]1
²
§metrics
^regularization_losses
¨non_trainable_variables
_trainable_variables
©layers
ªlayer_metrics
`	variables
 «layer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

¬0
­1
 
*
"0
#1
52
63
R4
S5
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 
 
 
 

"0
#1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

50
61
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

R0
S1
 
 
 
 
 
 
 
 
 
 
 
 
 
8

®total

¯count
°	variables
±	keras_api
I

²total

³count
´
_fn_kwargs
µ	variables
¶	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

®0
¯1

°	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

²0
³1

µ	variables

VARIABLE_VALUESGD/conv2d_26/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv2d_26/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)SGD/batch_normalization_38/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_38/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv2d_27/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv2d_27/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)SGD/batch_normalization_39/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_39/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_38/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_38/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_39/kernel/momentumYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_39/bias/momentumWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)SGD/batch_normalization_40/gamma/momentumXlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_40/beta/momentumWlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_40/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_40/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

serving_default_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿô

StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv2d_26/kernelconv2d_26/biasbatch_normalization_38/gammabatch_normalization_38/beta"batch_normalization_38/moving_mean&batch_normalization_38/moving_varianceconv2d_27/kernelconv2d_27/biasbatch_normalization_39/gammabatch_normalization_39/beta"batch_normalization_39/moving_mean&batch_normalization_39/moving_variancedense_38/kerneldense_38/biasdense_39/kerneldense_39/bias&batch_normalization_40/moving_variancebatch_normalization_40/gamma"batch_normalization_40/moving_meanbatch_normalization_40/betadense_40/kerneldense_40/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *-
f(R&
$__inference_signature_wrapper_266183
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp0batch_normalization_38/gamma/Read/ReadVariableOp/batch_normalization_38/beta/Read/ReadVariableOp6batch_normalization_38/moving_mean/Read/ReadVariableOp:batch_normalization_38/moving_variance/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp0batch_normalization_39/gamma/Read/ReadVariableOp/batch_normalization_39/beta/Read/ReadVariableOp6batch_normalization_39/moving_mean/Read/ReadVariableOp:batch_normalization_39/moving_variance/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOp0batch_normalization_40/gamma/Read/ReadVariableOp/batch_normalization_40/beta/Read/ReadVariableOp6batch_normalization_40/moving_mean/Read/ReadVariableOp:batch_normalization_40/moving_variance/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp1SGD/conv2d_26/kernel/momentum/Read/ReadVariableOp/SGD/conv2d_26/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_38/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_38/beta/momentum/Read/ReadVariableOp1SGD/conv2d_27/kernel/momentum/Read/ReadVariableOp/SGD/conv2d_27/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_39/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_39/beta/momentum/Read/ReadVariableOp0SGD/dense_38/kernel/momentum/Read/ReadVariableOp.SGD/dense_38/bias/momentum/Read/ReadVariableOp0SGD/dense_39/kernel/momentum/Read/ReadVariableOp.SGD/dense_39/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_40/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_40/beta/momentum/Read/ReadVariableOp0SGD/dense_40/kernel/momentum/Read/ReadVariableOp.SGD/dense_40/bias/momentum/Read/ReadVariableOpConst*;
Tin4
220	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *(
f#R!
__inference__traced_save_267317
ý
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_26/kernelconv2d_26/biasbatch_normalization_38/gammabatch_normalization_38/beta"batch_normalization_38/moving_mean&batch_normalization_38/moving_varianceconv2d_27/kernelconv2d_27/biasbatch_normalization_39/gammabatch_normalization_39/beta"batch_normalization_39/moving_mean&batch_normalization_39/moving_variancedense_38/kerneldense_38/biasdense_39/kerneldense_39/biasbatch_normalization_40/gammabatch_normalization_40/beta"batch_normalization_40/moving_mean&batch_normalization_40/moving_variancedense_40/kerneldense_40/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1SGD/conv2d_26/kernel/momentumSGD/conv2d_26/bias/momentum)SGD/batch_normalization_38/gamma/momentum(SGD/batch_normalization_38/beta/momentumSGD/conv2d_27/kernel/momentumSGD/conv2d_27/bias/momentum)SGD/batch_normalization_39/gamma/momentum(SGD/batch_normalization_39/beta/momentumSGD/dense_38/kernel/momentumSGD/dense_38/bias/momentumSGD/dense_39/kernel/momentumSGD/dense_39/bias/momentum)SGD/batch_normalization_40/gamma/momentum(SGD/batch_normalization_40/beta/momentumSGD/dense_40/kernel/momentumSGD/dense_40/bias/momentum*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *+
f&R$
"__inference__traced_restore_267465ò£

l
P__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_265052

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
ª
7__inference_batch_normalization_38_layer_call_fn_266653

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_2650042
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
G
+__inference_flatten_13_layer_call_fn_266952

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_flatten_13_layer_call_and_return_conditional_losses_2655912
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

º
.__inference_functional_27_layer_call_fn_266570

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_functional_27_layer_call_and_return_conditional_losses_2660552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
Ì&
¯
D__inference_dense_38_layer_call_and_return_conditional_losses_266932

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddÈ
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mull
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
á
~
)__inference_dense_40_layer_call_fn_267112

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2657072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266704

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿd:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
å
­
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266753

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÐ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_27_layer_call_fn_266762

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2654432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_267077

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

p
__inference_loss_fn_0_267123?
;conv2d_26_kernel_regularizer_square_readvariableop_resource
identityí
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_26_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulg
IdentityIdentity$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
à
ª
7__inference_batch_normalization_38_layer_call_fn_266717

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_2653712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ü
ª
I__inference_functional_27_layer_call_and_return_conditional_losses_266341

inputs,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource2
.batch_normalization_38_readvariableop_resource4
0batch_normalization_38_readvariableop_1_resourceC
?batch_normalization_38_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource2
.batch_normalization_39_readvariableop_resource4
0batch_normalization_39_readvariableop_1_resourceC
?batch_normalization_39_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource.
*dense_38_tensordot_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource1
-batch_normalization_40_assignmovingavg_2662773
/batch_normalization_40_assignmovingavg_1_266283@
<batch_normalization_40_batchnorm_mul_readvariableop_resource<
8batch_normalization_40_batchnorm_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource
identity¢%batch_normalization_38/AssignNewValue¢'batch_normalization_38/AssignNewValue_1¢%batch_normalization_39/AssignNewValue¢'batch_normalization_39/AssignNewValue_1¢:batch_normalization_40/AssignMovingAvg/AssignSubVariableOp¢<batch_normalization_40/AssignMovingAvg_1/AssignSubVariableOpÅ
average_pooling2d_39/AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
ksize
*
paddingVALID*
strides
2
average_pooling2d_39/AvgPool´
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOpá
conv2d_26/Conv2DConv2D%average_pooling2d_39/AvgPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingSAME*
strides
2
conv2d_26/Conv2D«
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp±
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
conv2d_26/Reluº
%batch_normalization_38/ReadVariableOpReadVariableOp.batch_normalization_38_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_38/ReadVariableOpÀ
'batch_normalization_38/ReadVariableOp_1ReadVariableOp0batch_normalization_38_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_38/ReadVariableOp_1í
6batch_normalization_38/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_38_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_38/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1ý
'batch_normalization_38/FusedBatchNormV3FusedBatchNormV3conv2d_26/Relu:activations:0-batch_normalization_38/ReadVariableOp:value:0/batch_normalization_38/ReadVariableOp_1:value:0>batch_normalization_38/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿd:::::*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_38/FusedBatchNormV3
%batch_normalization_38/AssignNewValueAssignVariableOp?batch_normalization_38_fusedbatchnormv3_readvariableop_resource4batch_normalization_38/FusedBatchNormV3:batch_mean:07^batch_normalization_38/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_38/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_38/AssignNewValue
'batch_normalization_38/AssignNewValue_1AssignVariableOpAbatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_38/FusedBatchNormV3:batch_variance:09^batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_38/AssignNewValue_1ë
average_pooling2d_40/AvgPoolAvgPool+batch_normalization_38/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_40/AvgPool´
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_27/Conv2D/ReadVariableOpà
conv2d_27/Conv2DConv2D%average_pooling2d_40/AvgPool:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_27/Conv2Dª
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp°
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_27/Relu¹
%batch_normalization_39/ReadVariableOpReadVariableOp.batch_normalization_39_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_39/ReadVariableOp¿
'batch_normalization_39/ReadVariableOp_1ReadVariableOp0batch_normalization_39_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_39/ReadVariableOp_1ì
6batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_39/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ø
'batch_normalization_39/FusedBatchNormV3FusedBatchNormV3conv2d_27/Relu:activations:0-batch_normalization_39/ReadVariableOp:value:0/batch_normalization_39/ReadVariableOp_1:value:0>batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2)
'batch_normalization_39/FusedBatchNormV3
%batch_normalization_39/AssignNewValueAssignVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource4batch_normalization_39/FusedBatchNormV3:batch_mean:07^batch_normalization_39/FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@batch_normalization_39/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_39/AssignNewValue
'batch_normalization_39/AssignNewValue_1AssignVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_39/FusedBatchNormV3:batch_variance:09^batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1*T
_classJ
HFloc:@batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_39/AssignNewValue_1ê
average_pooling2d_41/AvgPoolAvgPool+batch_normalization_39/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_41/AvgPool±
!dense_38/Tensordot/ReadVariableOpReadVariableOp*dense_38_tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype02#
!dense_38/Tensordot/ReadVariableOp|
dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_38/Tensordot/axes
dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_38/Tensordot/free
dense_38/Tensordot/ShapeShape%average_pooling2d_41/AvgPool:output:0*
T0*
_output_shapes
:2
dense_38/Tensordot/Shape
 dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/GatherV2/axisþ
dense_38/Tensordot/GatherV2GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/free:output:0)dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2
"dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_38/Tensordot/GatherV2_1/axis
dense_38/Tensordot/GatherV2_1GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/axes:output:0+dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2_1~
dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const¤
dense_38/Tensordot/ProdProd$dense_38/Tensordot/GatherV2:output:0!dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod
dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_1¬
dense_38/Tensordot/Prod_1Prod&dense_38/Tensordot/GatherV2_1:output:0#dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod_1
dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_38/Tensordot/concat/axisÝ
dense_38/Tensordot/concatConcatV2 dense_38/Tensordot/free:output:0 dense_38/Tensordot/axes:output:0'dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat°
dense_38/Tensordot/stackPack dense_38/Tensordot/Prod:output:0"dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/stackÎ
dense_38/Tensordot/transpose	Transpose%average_pooling2d_41/AvgPool:output:0"dense_38/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_38/Tensordot/transposeÃ
dense_38/Tensordot/ReshapeReshape dense_38/Tensordot/transpose:y:0!dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_38/Tensordot/ReshapeÂ
dense_38/Tensordot/MatMulMatMul#dense_38/Tensordot/Reshape:output:0)dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_38/Tensordot/MatMul
dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2
dense_38/Tensordot/Const_2
 dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/concat_1/axisê
dense_38/Tensordot/concat_1ConcatV2$dense_38/Tensordot/GatherV2:output:0#dense_38/Tensordot/Const_2:output:0)dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat_1¸
dense_38/TensordotReshape#dense_38/Tensordot/MatMul:product:0$dense_38/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_38/Tensordot§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_38/BiasAdd/ReadVariableOp¯
dense_38/BiasAddBiasAdddense_38/Tensordot:output:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_38/BiasAddu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÐ  2
flatten_13/Const
flatten_13/ReshapeReshapedense_38/BiasAdd:output:0flatten_13/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
flatten_13/Reshape©
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02 
dense_39/MatMul/ReadVariableOp£
dense_39/MatMulMatMulflatten_13/Reshape:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_39/MatMul§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp¥
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_39/BiasAdd¸
5batch_normalization_40/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_40/moments/mean/reduction_indicesç
#batch_normalization_40/moments/meanMeandense_39/BiasAdd:output:0>batch_normalization_40/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_40/moments/meanÁ
+batch_normalization_40/moments/StopGradientStopGradient,batch_normalization_40/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_40/moments/StopGradientü
0batch_normalization_40/moments/SquaredDifferenceSquaredDifferencedense_39/BiasAdd:output:04batch_normalization_40/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_40/moments/SquaredDifferenceÀ
9batch_normalization_40/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_40/moments/variance/reduction_indices
'batch_normalization_40/moments/varianceMean4batch_normalization_40/moments/SquaredDifference:z:0Bbatch_normalization_40/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_40/moments/varianceÅ
&batch_normalization_40/moments/SqueezeSqueeze,batch_normalization_40/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_40/moments/SqueezeÍ
(batch_normalization_40/moments/Squeeze_1Squeeze0batch_normalization_40/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_40/moments/Squeeze_1ã
,batch_normalization_40/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_40/AssignMovingAvg/266277*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_40/AssignMovingAvg/decayØ
5batch_normalization_40/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_40_assignmovingavg_266277*
_output_shapes
:*
dtype027
5batch_normalization_40/AssignMovingAvg/ReadVariableOp¶
*batch_normalization_40/AssignMovingAvg/subSub=batch_normalization_40/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_40/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_40/AssignMovingAvg/266277*
_output_shapes
:2,
*batch_normalization_40/AssignMovingAvg/sub­
*batch_normalization_40/AssignMovingAvg/mulMul.batch_normalization_40/AssignMovingAvg/sub:z:05batch_normalization_40/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_40/AssignMovingAvg/266277*
_output_shapes
:2,
*batch_normalization_40/AssignMovingAvg/mul
:batch_normalization_40/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_40_assignmovingavg_266277.batch_normalization_40/AssignMovingAvg/mul:z:06^batch_normalization_40/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_40/AssignMovingAvg/266277*
_output_shapes
 *
dtype02<
:batch_normalization_40/AssignMovingAvg/AssignSubVariableOpé
.batch_normalization_40/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_40/AssignMovingAvg_1/266283*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_40/AssignMovingAvg_1/decayÞ
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_40_assignmovingavg_1_266283*
_output_shapes
:*
dtype029
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOpÀ
,batch_normalization_40/AssignMovingAvg_1/subSub?batch_normalization_40/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_40/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_40/AssignMovingAvg_1/266283*
_output_shapes
:2.
,batch_normalization_40/AssignMovingAvg_1/sub·
,batch_normalization_40/AssignMovingAvg_1/mulMul0batch_normalization_40/AssignMovingAvg_1/sub:z:07batch_normalization_40/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_40/AssignMovingAvg_1/266283*
_output_shapes
:2.
,batch_normalization_40/AssignMovingAvg_1/mul
<batch_normalization_40/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_40_assignmovingavg_1_2662830batch_normalization_40/AssignMovingAvg_1/mul:z:08^batch_normalization_40/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_40/AssignMovingAvg_1/266283*
_output_shapes
 *
dtype02>
<batch_normalization_40/AssignMovingAvg_1/AssignSubVariableOp
&batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_40/batchnorm/add/yÞ
$batch_normalization_40/batchnorm/addAddV21batch_normalization_40/moments/Squeeze_1:output:0/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/add¨
&batch_normalization_40/batchnorm/RsqrtRsqrt(batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/Rsqrtã
3batch_normalization_40/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_40_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_40/batchnorm/mul/ReadVariableOpá
$batch_normalization_40/batchnorm/mulMul*batch_normalization_40/batchnorm/Rsqrt:y:0;batch_normalization_40/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/mulÎ
&batch_normalization_40/batchnorm/mul_1Muldense_39/BiasAdd:output:0(batch_normalization_40/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/mul_1×
&batch_normalization_40/batchnorm/mul_2Mul/batch_normalization_40/moments/Squeeze:output:0(batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/mul_2×
/batch_normalization_40/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_40_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_40/batchnorm/ReadVariableOpÝ
$batch_normalization_40/batchnorm/subSub7batch_normalization_40/batchnorm/ReadVariableOp:value:0*batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/subá
&batch_normalization_40/batchnorm/add_1AddV2*batch_normalization_40/batchnorm/mul_1:z:0(batch_normalization_40/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/add_1y
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_12/dropout/Const¸
dropout_12/dropout/MulMul*batch_normalization_40/batchnorm/add_1:z:0!dropout_12/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_12/dropout/Mul
dropout_12/dropout/ShapeShape*batch_normalization_40/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_12/dropout/ShapeÕ
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_12/dropout/GreaterEqual/yê
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_12/dropout/GreaterEqual 
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_12/dropout/Cast¦
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_12/dropout/Mul_1¨
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_40/MatMul/ReadVariableOp¤
dense_40/MatMulMatMuldropout_12/dropout/Mul_1:z:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_40/MatMul§
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_40/BiasAdd/ReadVariableOp¥
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_40/BiasAdd|
dense_40/SoftmaxSoftmaxdense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_40/SoftmaxÚ
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulÚ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulÑ
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_38_tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mulÏ
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/mul
IdentityIdentitydense_40/Softmax:softmax:0&^batch_normalization_38/AssignNewValue(^batch_normalization_38/AssignNewValue_1&^batch_normalization_39/AssignNewValue(^batch_normalization_39/AssignNewValue_1;^batch_normalization_40/AssignMovingAvg/AssignSubVariableOp=^batch_normalization_40/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::2N
%batch_normalization_38/AssignNewValue%batch_normalization_38/AssignNewValue2R
'batch_normalization_38/AssignNewValue_1'batch_normalization_38/AssignNewValue_12N
%batch_normalization_39/AssignNewValue%batch_normalization_39/AssignNewValue2R
'batch_normalization_39/AssignNewValue_1'batch_normalization_39/AssignNewValue_12x
:batch_normalization_40/AssignMovingAvg/AssignSubVariableOp:batch_normalization_40/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_40/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_40/AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
å
­
E__inference_conv2d_27_layer_call_and_return_conditional_losses_265443

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
ReluÐ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/muln
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_265478

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ
¬
D__inference_dense_39_layer_call_and_return_conditional_losses_265615

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddÆ
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/muld
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs
è
­
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266593

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
ReluÐ
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¾
b
F__inference_flatten_13_layer_call_and_return_conditional_losses_265591

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÐ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ö
¯
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_265004

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266686

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿd:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ñ

R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266800

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_265151

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ã
~
)__inference_dense_39_layer_call_fn_266983

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2656152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs

~
)__inference_dense_38_layer_call_fn_266941

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_2655692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


*__inference_conv2d_26_layer_call_fn_266602

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_2653362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ü
ª
7__inference_batch_normalization_39_layer_call_fn_266813

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2654782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ïe
Ê
I__inference_functional_27_layer_call_and_return_conditional_losses_265748	
input
conv2d_26_265347
conv2d_26_265349!
batch_normalization_38_265416!
batch_normalization_38_265418!
batch_normalization_38_265420!
batch_normalization_38_265422
conv2d_27_265454
conv2d_27_265456!
batch_normalization_39_265523!
batch_normalization_39_265525!
batch_normalization_39_265527!
batch_normalization_39_265529
dense_38_265580
dense_38_265582
dense_39_265626
dense_39_265628!
batch_normalization_40_265657!
batch_normalization_40_265659!
batch_normalization_40_265661!
batch_normalization_40_265663
dense_40_265718
dense_40_265720
identity¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢.batch_normalization_40/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall¢!conv2d_27/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢"dropout_12/StatefulPartitionedCall
$average_pooling2d_39/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_39_layer_call_and_return_conditional_losses_2649362&
$average_pooling2d_39/PartitionedCallÎ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_39/PartitionedCall:output:0conv2d_26_265347conv2d_26_265349*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_2653362#
!conv2d_26/StatefulPartitionedCallÌ
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_38_265416batch_normalization_38_265418batch_normalization_38_265420batch_normalization_38_265422*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_26537120
.batch_normalization_38/StatefulPartitionedCall·
$average_pooling2d_40/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_2650522&
$average_pooling2d_40/PartitionedCallÍ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_40/PartitionedCall:output:0conv2d_27_265454conv2d_27_265456*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2654432#
!conv2d_27/StatefulPartitionedCallË
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_39_265523batch_normalization_39_265525batch_normalization_39_265527batch_normalization_39_265529*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_26547820
.batch_normalization_39/StatefulPartitionedCall¶
$average_pooling2d_41/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_2651682&
$average_pooling2d_41/PartitionedCallÈ
 dense_38/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_41/PartitionedCall:output:0dense_38_265580dense_38_265582*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_2655692"
 dense_38/StatefulPartitionedCall
flatten_13/PartitionedCallPartitionedCall)dense_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_flatten_13_layer_call_and_return_conditional_losses_2655912
flatten_13/PartitionedCall¶
 dense_39/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_39_265626dense_39_265628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2656152"
 dense_39/StatefulPartitionedCallÂ
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0batch_normalization_40_265657batch_normalization_40_265659batch_normalization_40_265661batch_normalization_40_265663*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_26527020
.batch_normalization_40/StatefulPartitionedCall¨
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2656782$
"dropout_12/StatefulPartitionedCall¾
 dense_40/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_40_265718dense_40_265720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2657072"
 dense_40/StatefulPartitionedCallÂ
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_265347*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulÂ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_265454*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul¶
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_265580*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul·
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_39_265626*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/mulæ
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall:W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô

_user_specified_nameinput
Ì&
¯
D__inference_dense_38_layer_call_and_return_conditional_losses_265569

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAddÈ
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mull
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª
ª
7__inference_batch_normalization_38_layer_call_fn_266666

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_2650352
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_265683

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤d
¦
I__inference_functional_27_layer_call_and_return_conditional_losses_266055

inputs
conv2d_26_265974
conv2d_26_265976!
batch_normalization_38_265979!
batch_normalization_38_265981!
batch_normalization_38_265983!
batch_normalization_38_265985
conv2d_27_265989
conv2d_27_265991!
batch_normalization_39_265994!
batch_normalization_39_265996!
batch_normalization_39_265998!
batch_normalization_39_266000
dense_38_266004
dense_38_266006
dense_39_266010
dense_39_266012!
batch_normalization_40_266015!
batch_normalization_40_266017!
batch_normalization_40_266019!
batch_normalization_40_266021
dense_40_266025
dense_40_266027
identity¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢.batch_normalization_40/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall¢!conv2d_27/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall
$average_pooling2d_39/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_39_layer_call_and_return_conditional_losses_2649362&
$average_pooling2d_39/PartitionedCallÎ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_39/PartitionedCall:output:0conv2d_26_265974conv2d_26_265976*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_2653362#
!conv2d_26/StatefulPartitionedCallÎ
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_38_265979batch_normalization_38_265981batch_normalization_38_265983batch_normalization_38_265985*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_26538920
.batch_normalization_38/StatefulPartitionedCall·
$average_pooling2d_40/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_2650522&
$average_pooling2d_40/PartitionedCallÍ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_40/PartitionedCall:output:0conv2d_27_265989conv2d_27_265991*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2654432#
!conv2d_27/StatefulPartitionedCallÍ
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_39_265994batch_normalization_39_265996batch_normalization_39_265998batch_normalization_39_266000*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_26549620
.batch_normalization_39/StatefulPartitionedCall¶
$average_pooling2d_41/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_2651682&
$average_pooling2d_41/PartitionedCallÈ
 dense_38/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_41/PartitionedCall:output:0dense_38_266004dense_38_266006*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_2655692"
 dense_38/StatefulPartitionedCall
flatten_13/PartitionedCallPartitionedCall)dense_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_flatten_13_layer_call_and_return_conditional_losses_2655912
flatten_13/PartitionedCall¶
 dense_39/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_39_266010dense_39_266012*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2656152"
 dense_39/StatefulPartitionedCallÄ
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0batch_normalization_40_266015batch_normalization_40_266017batch_normalization_40_266019batch_normalization_40_266021*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_26530320
.batch_normalization_40/StatefulPartitionedCall
dropout_12/PartitionedCallPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2656832
dropout_12/PartitionedCall¶
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_40_266025dense_40_266027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2657072"
 dense_40/StatefulPartitionedCallÂ
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_265974*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulÂ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_265989*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul¶
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_266004*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul·
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_39_266010*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/mulÁ
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs


R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_265303

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
ª
7__inference_batch_normalization_39_layer_call_fn_266890

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2651512
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

l
P__inference_average_pooling2d_39_layer_call_and_return_conditional_losses_264936

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 )
Ë
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_265270

inputs
assignmovingavg_265245
assignmovingavg_1_265251)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/265245*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_265245*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpÃ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/265245*
_output_shapes
:2
AssignMovingAvg/subº
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/265245*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_265245AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/265245*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/265251*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_265251*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÍ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/265251*
_output_shapes
:2
AssignMovingAvg_1/subÄ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/265251*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_265251AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/265251*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
­
E__inference_conv2d_26_layer_call_and_return_conditional_losses_265336

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
ReluÐ
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_265678

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

º
.__inference_functional_27_layer_call_fn_266521

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_functional_27_layer_call_and_return_conditional_losses_2659212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_265371

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿd:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ú

o
__inference_loss_fn_2_267145>
:dense_38_kernel_regularizer_square_readvariableop_resource
identityá
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_38_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mulf
IdentityIdentity#dense_38/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
Ñ

R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_265496

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¼
Q
5__inference_average_pooling2d_40_layer_call_fn_265058

inputs
identityö
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_2650522
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266782

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ
¬
D__inference_dense_39_layer_call_and_return_conditional_losses_266974

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddÆ
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/muld
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÐ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
 
_user_specified_nameinputs


R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_267039

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
ª
7__inference_batch_normalization_40_layer_call_fn_267052

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2652702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Òe
Ë
I__inference_functional_27_layer_call_and_return_conditional_losses_265921

inputs
conv2d_26_265840
conv2d_26_265842!
batch_normalization_38_265845!
batch_normalization_38_265847!
batch_normalization_38_265849!
batch_normalization_38_265851
conv2d_27_265855
conv2d_27_265857!
batch_normalization_39_265860!
batch_normalization_39_265862!
batch_normalization_39_265864!
batch_normalization_39_265866
dense_38_265870
dense_38_265872
dense_39_265876
dense_39_265878!
batch_normalization_40_265881!
batch_normalization_40_265883!
batch_normalization_40_265885!
batch_normalization_40_265887
dense_40_265891
dense_40_265893
identity¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢.batch_normalization_40/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall¢!conv2d_27/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall¢"dropout_12/StatefulPartitionedCall
$average_pooling2d_39/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_39_layer_call_and_return_conditional_losses_2649362&
$average_pooling2d_39/PartitionedCallÎ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_39/PartitionedCall:output:0conv2d_26_265840conv2d_26_265842*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_2653362#
!conv2d_26/StatefulPartitionedCallÌ
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_38_265845batch_normalization_38_265847batch_normalization_38_265849batch_normalization_38_265851*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_26537120
.batch_normalization_38/StatefulPartitionedCall·
$average_pooling2d_40/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_2650522&
$average_pooling2d_40/PartitionedCallÍ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_40/PartitionedCall:output:0conv2d_27_265855conv2d_27_265857*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2654432#
!conv2d_27/StatefulPartitionedCallË
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_39_265860batch_normalization_39_265862batch_normalization_39_265864batch_normalization_39_265866*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_26547820
.batch_normalization_39/StatefulPartitionedCall¶
$average_pooling2d_41/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_2651682&
$average_pooling2d_41/PartitionedCallÈ
 dense_38/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_41/PartitionedCall:output:0dense_38_265870dense_38_265872*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_2655692"
 dense_38/StatefulPartitionedCall
flatten_13/PartitionedCallPartitionedCall)dense_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_flatten_13_layer_call_and_return_conditional_losses_2655912
flatten_13/PartitionedCall¶
 dense_39/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_39_265876dense_39_265878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2656152"
 dense_39/StatefulPartitionedCallÂ
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0batch_normalization_40_265881batch_normalization_40_265883batch_normalization_40_265885batch_normalization_40_265887*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_26527020
.batch_normalization_40/StatefulPartitionedCall¨
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2656782$
"dropout_12/StatefulPartitionedCall¾
 dense_40/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_40_265891dense_40_265893*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2657072"
 dense_40/StatefulPartitionedCallÂ
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_265840*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulÂ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_265855*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul¶
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_265870*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul·
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_39_265876*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/mulæ
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_265035

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266846

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¼
Q
5__inference_average_pooling2d_39_layer_call_fn_264942

inputs
identityö
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_39_layer_call_and_return_conditional_losses_2649362
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
d
+__inference_dropout_12_layer_call_fn_267087

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2656782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

o
__inference_loss_fn_3_267156>
:dense_39_kernel_regularizer_square_readvariableop_resource
identityâ
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_39_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/mulf
IdentityIdentity#dense_39/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
¼
Q
5__inference_average_pooling2d_41_layer_call_fn_265174

inputs
identityö
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_2651682
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266640

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
ª
7__inference_batch_normalization_39_layer_call_fn_266826

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2654962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¹
.__inference_functional_27_layer_call_fn_266102	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_functional_27_layer_call_and_return_conditional_losses_2660552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô

_user_specified_nameinput
Ö
¯
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266622

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
¢

I__inference_functional_27_layer_call_and_return_conditional_losses_266472

inputs,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource2
.batch_normalization_38_readvariableop_resource4
0batch_normalization_38_readvariableop_1_resourceC
?batch_normalization_38_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource2
.batch_normalization_39_readvariableop_resource4
0batch_normalization_39_readvariableop_1_resourceC
?batch_normalization_39_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource.
*dense_38_tensordot_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource<
8batch_normalization_40_batchnorm_readvariableop_resource@
<batch_normalization_40_batchnorm_mul_readvariableop_resource>
:batch_normalization_40_batchnorm_readvariableop_1_resource>
:batch_normalization_40_batchnorm_readvariableop_2_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource
identityÅ
average_pooling2d_39/AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
ksize
*
paddingVALID*
strides
2
average_pooling2d_39/AvgPool´
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_26/Conv2D/ReadVariableOpá
conv2d_26/Conv2DConv2D%average_pooling2d_39/AvgPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingSAME*
strides
2
conv2d_26/Conv2D«
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp±
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
conv2d_26/BiasAdd
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
conv2d_26/Reluº
%batch_normalization_38/ReadVariableOpReadVariableOp.batch_normalization_38_readvariableop_resource*
_output_shapes	
:*
dtype02'
%batch_normalization_38/ReadVariableOpÀ
'batch_normalization_38/ReadVariableOp_1ReadVariableOp0batch_normalization_38_readvariableop_1_resource*
_output_shapes	
:*
dtype02)
'batch_normalization_38/ReadVariableOp_1í
6batch_normalization_38/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_38_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype028
6batch_normalization_38/FusedBatchNormV3/ReadVariableOpó
8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02:
8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1ï
'batch_normalization_38/FusedBatchNormV3FusedBatchNormV3conv2d_26/Relu:activations:0-batch_normalization_38/ReadVariableOp:value:0/batch_normalization_38/ReadVariableOp_1:value:0>batch_normalization_38/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿd:::::*
epsilon%o:*
is_training( 2)
'batch_normalization_38/FusedBatchNormV3ë
average_pooling2d_40/AvgPoolAvgPool+batch_normalization_38/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_40/AvgPool´
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_27/Conv2D/ReadVariableOpà
conv2d_27/Conv2DConv2D%average_pooling2d_40/AvgPool:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_27/Conv2Dª
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp°
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_27/Relu¹
%batch_normalization_39/ReadVariableOpReadVariableOp.batch_normalization_39_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_39/ReadVariableOp¿
'batch_normalization_39/ReadVariableOp_1ReadVariableOp0batch_normalization_39_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_39/ReadVariableOp_1ì
6batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_39/FusedBatchNormV3/ReadVariableOpò
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ê
'batch_normalization_39/FusedBatchNormV3FusedBatchNormV3conv2d_27/Relu:activations:0-batch_normalization_39/ReadVariableOp:value:0/batch_normalization_39/ReadVariableOp_1:value:0>batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2)
'batch_normalization_39/FusedBatchNormV3ê
average_pooling2d_41/AvgPoolAvgPool+batch_normalization_39/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_41/AvgPool±
!dense_38/Tensordot/ReadVariableOpReadVariableOp*dense_38_tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype02#
!dense_38/Tensordot/ReadVariableOp|
dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_38/Tensordot/axes
dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_38/Tensordot/free
dense_38/Tensordot/ShapeShape%average_pooling2d_41/AvgPool:output:0*
T0*
_output_shapes
:2
dense_38/Tensordot/Shape
 dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/GatherV2/axisþ
dense_38/Tensordot/GatherV2GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/free:output:0)dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2
"dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_38/Tensordot/GatherV2_1/axis
dense_38/Tensordot/GatherV2_1GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/axes:output:0+dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2_1~
dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const¤
dense_38/Tensordot/ProdProd$dense_38/Tensordot/GatherV2:output:0!dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod
dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_1¬
dense_38/Tensordot/Prod_1Prod&dense_38/Tensordot/GatherV2_1:output:0#dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod_1
dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_38/Tensordot/concat/axisÝ
dense_38/Tensordot/concatConcatV2 dense_38/Tensordot/free:output:0 dense_38/Tensordot/axes:output:0'dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat°
dense_38/Tensordot/stackPack dense_38/Tensordot/Prod:output:0"dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/stackÎ
dense_38/Tensordot/transpose	Transpose%average_pooling2d_41/AvgPool:output:0"dense_38/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_38/Tensordot/transposeÃ
dense_38/Tensordot/ReshapeReshape dense_38/Tensordot/transpose:y:0!dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_38/Tensordot/ReshapeÂ
dense_38/Tensordot/MatMulMatMul#dense_38/Tensordot/Reshape:output:0)dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_38/Tensordot/MatMul
dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2
dense_38/Tensordot/Const_2
 dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/concat_1/axisê
dense_38/Tensordot/concat_1ConcatV2$dense_38/Tensordot/GatherV2:output:0#dense_38/Tensordot/Const_2:output:0)dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat_1¸
dense_38/TensordotReshape#dense_38/Tensordot/MatMul:product:0$dense_38/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_38/Tensordot§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_38/BiasAdd/ReadVariableOp¯
dense_38/BiasAddBiasAdddense_38/Tensordot:output:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_38/BiasAddu
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÐ  2
flatten_13/Const
flatten_13/ReshapeReshapedense_38/BiasAdd:output:0flatten_13/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
flatten_13/Reshape©
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02 
dense_39/MatMul/ReadVariableOp£
dense_39/MatMulMatMulflatten_13/Reshape:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_39/MatMul§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp¥
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_39/BiasAdd×
/batch_normalization_40/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_40_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype021
/batch_normalization_40/batchnorm/ReadVariableOp
&batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2(
&batch_normalization_40/batchnorm/add/yä
$batch_normalization_40/batchnorm/addAddV27batch_normalization_40/batchnorm/ReadVariableOp:value:0/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/add¨
&batch_normalization_40/batchnorm/RsqrtRsqrt(batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/Rsqrtã
3batch_normalization_40/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_40_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization_40/batchnorm/mul/ReadVariableOpá
$batch_normalization_40/batchnorm/mulMul*batch_normalization_40/batchnorm/Rsqrt:y:0;batch_normalization_40/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/mulÎ
&batch_normalization_40/batchnorm/mul_1Muldense_39/BiasAdd:output:0(batch_normalization_40/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/mul_1Ý
1batch_normalization_40/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_40_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype023
1batch_normalization_40/batchnorm/ReadVariableOp_1á
&batch_normalization_40/batchnorm/mul_2Mul9batch_normalization_40/batchnorm/ReadVariableOp_1:value:0(batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/mul_2Ý
1batch_normalization_40/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_40_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype023
1batch_normalization_40/batchnorm/ReadVariableOp_2ß
$batch_normalization_40/batchnorm/subSub9batch_normalization_40/batchnorm/ReadVariableOp_2:value:0*batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/subá
&batch_normalization_40/batchnorm/add_1AddV2*batch_normalization_40/batchnorm/mul_1:z:0(batch_normalization_40/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/add_1
dropout_12/IdentityIdentity*batch_normalization_40/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_12/Identity¨
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_40/MatMul/ReadVariableOp¤
dense_40/MatMulMatMuldropout_12/Identity:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_40/MatMul§
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_40/BiasAdd/ReadVariableOp¥
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_40/BiasAdd|
dense_40/SoftmaxSoftmaxdense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_40/SoftmaxÚ
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulÚ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulÑ
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_38_tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mulÏ
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/muln
IdentityIdentitydense_40/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô:::::::::::::::::::::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
â
ª
7__inference_batch_normalization_38_layer_call_fn_266730

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_2653892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¤
ª
7__inference_batch_normalization_39_layer_call_fn_266877

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2651202
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾
b
F__inference_flatten_13_layer_call_and_return_conditional_losses_266947

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÐ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

G
+__inference_dropout_12_layer_call_fn_267092

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2656832
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

l
P__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_265168

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_267082

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 )
Ë
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_267019

inputs
assignmovingavg_266994
assignmovingavg_1_267000)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/266994*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_266994*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpÃ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/266994*
_output_shapes
:2
AssignMovingAvg/subº
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/266994*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_266994AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/266994*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/267000*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_267000*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÍ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/267000*
_output_shapes
:2
AssignMovingAvg_1/subÄ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/267000*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_267000AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/267000*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266864

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú
­
!__inference__wrapped_model_264930	
input:
6functional_27_conv2d_26_conv2d_readvariableop_resource;
7functional_27_conv2d_26_biasadd_readvariableop_resource@
<functional_27_batch_normalization_38_readvariableop_resourceB
>functional_27_batch_normalization_38_readvariableop_1_resourceQ
Mfunctional_27_batch_normalization_38_fusedbatchnormv3_readvariableop_resourceS
Ofunctional_27_batch_normalization_38_fusedbatchnormv3_readvariableop_1_resource:
6functional_27_conv2d_27_conv2d_readvariableop_resource;
7functional_27_conv2d_27_biasadd_readvariableop_resource@
<functional_27_batch_normalization_39_readvariableop_resourceB
>functional_27_batch_normalization_39_readvariableop_1_resourceQ
Mfunctional_27_batch_normalization_39_fusedbatchnormv3_readvariableop_resourceS
Ofunctional_27_batch_normalization_39_fusedbatchnormv3_readvariableop_1_resource<
8functional_27_dense_38_tensordot_readvariableop_resource:
6functional_27_dense_38_biasadd_readvariableop_resource9
5functional_27_dense_39_matmul_readvariableop_resource:
6functional_27_dense_39_biasadd_readvariableop_resourceJ
Ffunctional_27_batch_normalization_40_batchnorm_readvariableop_resourceN
Jfunctional_27_batch_normalization_40_batchnorm_mul_readvariableop_resourceL
Hfunctional_27_batch_normalization_40_batchnorm_readvariableop_1_resourceL
Hfunctional_27_batch_normalization_40_batchnorm_readvariableop_2_resource9
5functional_27_dense_40_matmul_readvariableop_resource:
6functional_27_dense_40_biasadd_readvariableop_resource
identityà
*functional_27/average_pooling2d_39/AvgPoolAvgPoolinput*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
ksize
*
paddingVALID*
strides
2,
*functional_27/average_pooling2d_39/AvgPoolÞ
-functional_27/conv2d_26/Conv2D/ReadVariableOpReadVariableOp6functional_27_conv2d_26_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02/
-functional_27/conv2d_26/Conv2D/ReadVariableOp
functional_27/conv2d_26/Conv2DConv2D3functional_27/average_pooling2d_39/AvgPool:output:05functional_27/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
paddingSAME*
strides
2 
functional_27/conv2d_26/Conv2DÕ
.functional_27/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp7functional_27_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.functional_27/conv2d_26/BiasAdd/ReadVariableOpé
functional_27/conv2d_26/BiasAddBiasAdd'functional_27/conv2d_26/Conv2D:output:06functional_27/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2!
functional_27/conv2d_26/BiasAdd©
functional_27/conv2d_26/ReluRelu(functional_27/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
functional_27/conv2d_26/Reluä
3functional_27/batch_normalization_38/ReadVariableOpReadVariableOp<functional_27_batch_normalization_38_readvariableop_resource*
_output_shapes	
:*
dtype025
3functional_27/batch_normalization_38/ReadVariableOpê
5functional_27/batch_normalization_38/ReadVariableOp_1ReadVariableOp>functional_27_batch_normalization_38_readvariableop_1_resource*
_output_shapes	
:*
dtype027
5functional_27/batch_normalization_38/ReadVariableOp_1
Dfunctional_27/batch_normalization_38/FusedBatchNormV3/ReadVariableOpReadVariableOpMfunctional_27_batch_normalization_38_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02F
Dfunctional_27/batch_normalization_38/FusedBatchNormV3/ReadVariableOp
Ffunctional_27/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfunctional_27_batch_normalization_38_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02H
Ffunctional_27/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1Ñ
5functional_27/batch_normalization_38/FusedBatchNormV3FusedBatchNormV3*functional_27/conv2d_26/Relu:activations:0;functional_27/batch_normalization_38/ReadVariableOp:value:0=functional_27/batch_normalization_38/ReadVariableOp_1:value:0Lfunctional_27/batch_normalization_38/FusedBatchNormV3/ReadVariableOp:value:0Nfunctional_27/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿd:::::*
epsilon%o:*
is_training( 27
5functional_27/batch_normalization_38/FusedBatchNormV3
*functional_27/average_pooling2d_40/AvgPoolAvgPool9functional_27/batch_normalization_38/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2,
*functional_27/average_pooling2d_40/AvgPoolÞ
-functional_27/conv2d_27/Conv2D/ReadVariableOpReadVariableOp6functional_27_conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02/
-functional_27/conv2d_27/Conv2D/ReadVariableOp
functional_27/conv2d_27/Conv2DConv2D3functional_27/average_pooling2d_40/AvgPool:output:05functional_27/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2 
functional_27/conv2d_27/Conv2DÔ
.functional_27/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp7functional_27_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.functional_27/conv2d_27/BiasAdd/ReadVariableOpè
functional_27/conv2d_27/BiasAddBiasAdd'functional_27/conv2d_27/Conv2D:output:06functional_27/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
functional_27/conv2d_27/BiasAdd¨
functional_27/conv2d_27/ReluRelu(functional_27/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
functional_27/conv2d_27/Reluã
3functional_27/batch_normalization_39/ReadVariableOpReadVariableOp<functional_27_batch_normalization_39_readvariableop_resource*
_output_shapes
:@*
dtype025
3functional_27/batch_normalization_39/ReadVariableOpé
5functional_27/batch_normalization_39/ReadVariableOp_1ReadVariableOp>functional_27_batch_normalization_39_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5functional_27/batch_normalization_39/ReadVariableOp_1
Dfunctional_27/batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOpMfunctional_27_batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dfunctional_27/batch_normalization_39/FusedBatchNormV3/ReadVariableOp
Ffunctional_27/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfunctional_27_batch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Ffunctional_27/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1Ì
5functional_27/batch_normalization_39/FusedBatchNormV3FusedBatchNormV3*functional_27/conv2d_27/Relu:activations:0;functional_27/batch_normalization_39/ReadVariableOp:value:0=functional_27/batch_normalization_39/ReadVariableOp_1:value:0Lfunctional_27/batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0Nfunctional_27/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 27
5functional_27/batch_normalization_39/FusedBatchNormV3
*functional_27/average_pooling2d_41/AvgPoolAvgPool9functional_27/batch_normalization_39/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2,
*functional_27/average_pooling2d_41/AvgPoolÛ
/functional_27/dense_38/Tensordot/ReadVariableOpReadVariableOp8functional_27_dense_38_tensordot_readvariableop_resource*
_output_shapes

:@d*
dtype021
/functional_27/dense_38/Tensordot/ReadVariableOp
%functional_27/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%functional_27/dense_38/Tensordot/axes£
%functional_27/dense_38/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%functional_27/dense_38/Tensordot/free³
&functional_27/dense_38/Tensordot/ShapeShape3functional_27/average_pooling2d_41/AvgPool:output:0*
T0*
_output_shapes
:2(
&functional_27/dense_38/Tensordot/Shape¢
.functional_27/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_27/dense_38/Tensordot/GatherV2/axisÄ
)functional_27/dense_38/Tensordot/GatherV2GatherV2/functional_27/dense_38/Tensordot/Shape:output:0.functional_27/dense_38/Tensordot/free:output:07functional_27/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)functional_27/dense_38/Tensordot/GatherV2¦
0functional_27/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0functional_27/dense_38/Tensordot/GatherV2_1/axisÊ
+functional_27/dense_38/Tensordot/GatherV2_1GatherV2/functional_27/dense_38/Tensordot/Shape:output:0.functional_27/dense_38/Tensordot/axes:output:09functional_27/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+functional_27/dense_38/Tensordot/GatherV2_1
&functional_27/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&functional_27/dense_38/Tensordot/ConstÜ
%functional_27/dense_38/Tensordot/ProdProd2functional_27/dense_38/Tensordot/GatherV2:output:0/functional_27/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%functional_27/dense_38/Tensordot/Prod
(functional_27/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(functional_27/dense_38/Tensordot/Const_1ä
'functional_27/dense_38/Tensordot/Prod_1Prod4functional_27/dense_38/Tensordot/GatherV2_1:output:01functional_27/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'functional_27/dense_38/Tensordot/Prod_1
,functional_27/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,functional_27/dense_38/Tensordot/concat/axis£
'functional_27/dense_38/Tensordot/concatConcatV2.functional_27/dense_38/Tensordot/free:output:0.functional_27/dense_38/Tensordot/axes:output:05functional_27/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'functional_27/dense_38/Tensordot/concatè
&functional_27/dense_38/Tensordot/stackPack.functional_27/dense_38/Tensordot/Prod:output:00functional_27/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&functional_27/dense_38/Tensordot/stack
*functional_27/dense_38/Tensordot/transpose	Transpose3functional_27/average_pooling2d_41/AvgPool:output:00functional_27/dense_38/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_27/dense_38/Tensordot/transposeû
(functional_27/dense_38/Tensordot/ReshapeReshape.functional_27/dense_38/Tensordot/transpose:y:0/functional_27/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(functional_27/dense_38/Tensordot/Reshapeú
'functional_27/dense_38/Tensordot/MatMulMatMul1functional_27/dense_38/Tensordot/Reshape:output:07functional_27/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2)
'functional_27/dense_38/Tensordot/MatMul
(functional_27/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d2*
(functional_27/dense_38/Tensordot/Const_2¢
.functional_27/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.functional_27/dense_38/Tensordot/concat_1/axis°
)functional_27/dense_38/Tensordot/concat_1ConcatV22functional_27/dense_38/Tensordot/GatherV2:output:01functional_27/dense_38/Tensordot/Const_2:output:07functional_27/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)functional_27/dense_38/Tensordot/concat_1ð
 functional_27/dense_38/TensordotReshape1functional_27/dense_38/Tensordot/MatMul:product:02functional_27/dense_38/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2"
 functional_27/dense_38/TensordotÑ
-functional_27/dense_38/BiasAdd/ReadVariableOpReadVariableOp6functional_27_dense_38_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-functional_27/dense_38/BiasAdd/ReadVariableOpç
functional_27/dense_38/BiasAddBiasAdd)functional_27/dense_38/Tensordot:output:05functional_27/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2 
functional_27/dense_38/BiasAdd
functional_27/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÐ  2 
functional_27/flatten_13/ConstÔ
 functional_27/flatten_13/ReshapeReshape'functional_27/dense_38/BiasAdd:output:0'functional_27/flatten_13/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2"
 functional_27/flatten_13/ReshapeÓ
,functional_27/dense_39/MatMul/ReadVariableOpReadVariableOp5functional_27_dense_39_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02.
,functional_27/dense_39/MatMul/ReadVariableOpÛ
functional_27/dense_39/MatMulMatMul)functional_27/flatten_13/Reshape:output:04functional_27/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_27/dense_39/MatMulÑ
-functional_27/dense_39/BiasAdd/ReadVariableOpReadVariableOp6functional_27_dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_27/dense_39/BiasAdd/ReadVariableOpÝ
functional_27/dense_39/BiasAddBiasAdd'functional_27/dense_39/MatMul:product:05functional_27/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
functional_27/dense_39/BiasAdd
=functional_27/batch_normalization_40/batchnorm/ReadVariableOpReadVariableOpFfunctional_27_batch_normalization_40_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02?
=functional_27/batch_normalization_40/batchnorm/ReadVariableOp±
4functional_27/batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:26
4functional_27/batch_normalization_40/batchnorm/add/y
2functional_27/batch_normalization_40/batchnorm/addAddV2Efunctional_27/batch_normalization_40/batchnorm/ReadVariableOp:value:0=functional_27/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
:24
2functional_27/batch_normalization_40/batchnorm/addÒ
4functional_27/batch_normalization_40/batchnorm/RsqrtRsqrt6functional_27/batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
:26
4functional_27/batch_normalization_40/batchnorm/Rsqrt
Afunctional_27/batch_normalization_40/batchnorm/mul/ReadVariableOpReadVariableOpJfunctional_27_batch_normalization_40_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02C
Afunctional_27/batch_normalization_40/batchnorm/mul/ReadVariableOp
2functional_27/batch_normalization_40/batchnorm/mulMul8functional_27/batch_normalization_40/batchnorm/Rsqrt:y:0Ifunctional_27/batch_normalization_40/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:24
2functional_27/batch_normalization_40/batchnorm/mul
4functional_27/batch_normalization_40/batchnorm/mul_1Mul'functional_27/dense_39/BiasAdd:output:06functional_27/batch_normalization_40/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4functional_27/batch_normalization_40/batchnorm/mul_1
?functional_27/batch_normalization_40/batchnorm/ReadVariableOp_1ReadVariableOpHfunctional_27_batch_normalization_40_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?functional_27/batch_normalization_40/batchnorm/ReadVariableOp_1
4functional_27/batch_normalization_40/batchnorm/mul_2MulGfunctional_27/batch_normalization_40/batchnorm/ReadVariableOp_1:value:06functional_27/batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
:26
4functional_27/batch_normalization_40/batchnorm/mul_2
?functional_27/batch_normalization_40/batchnorm/ReadVariableOp_2ReadVariableOpHfunctional_27_batch_normalization_40_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02A
?functional_27/batch_normalization_40/batchnorm/ReadVariableOp_2
2functional_27/batch_normalization_40/batchnorm/subSubGfunctional_27/batch_normalization_40/batchnorm/ReadVariableOp_2:value:08functional_27/batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
:24
2functional_27/batch_normalization_40/batchnorm/sub
4functional_27/batch_normalization_40/batchnorm/add_1AddV28functional_27/batch_normalization_40/batchnorm/mul_1:z:06functional_27/batch_normalization_40/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4functional_27/batch_normalization_40/batchnorm/add_1¾
!functional_27/dropout_12/IdentityIdentity8functional_27/batch_normalization_40/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!functional_27/dropout_12/IdentityÒ
,functional_27/dense_40/MatMul/ReadVariableOpReadVariableOp5functional_27_dense_40_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,functional_27/dense_40/MatMul/ReadVariableOpÜ
functional_27/dense_40/MatMulMatMul*functional_27/dropout_12/Identity:output:04functional_27/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_27/dense_40/MatMulÑ
-functional_27/dense_40/BiasAdd/ReadVariableOpReadVariableOp6functional_27_dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_27/dense_40/BiasAdd/ReadVariableOpÝ
functional_27/dense_40/BiasAddBiasAdd'functional_27/dense_40/MatMul:product:05functional_27/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
functional_27/dense_40/BiasAdd¦
functional_27/dense_40/SoftmaxSoftmax'functional_27/dense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
functional_27/dense_40/Softmax|
IdentityIdentity(functional_27/dense_40/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô:::::::::::::::::::::::W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô

_user_specified_nameinput
Ý

R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_265389

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿd:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿd:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
±
¬
D__inference_dense_40_layer_call_and_return_conditional_losses_267103

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ª
7__inference_batch_normalization_40_layer_call_fn_267065

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2653032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¹
.__inference_functional_27_layer_call_fn_265968	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_functional_27_layer_call_and_return_conditional_losses_2659212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô

_user_specified_nameinput
±
¬
D__inference_dense_40_layer_call_and_return_conditional_losses_265707

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

p
__inference_loss_fn_1_267134?
;conv2d_27_kernel_regularizer_square_readvariableop_resource
identityí
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_27_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mulg
IdentityIdentity$conv2d_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
¡d
¥
I__inference_functional_27_layer_call_and_return_conditional_losses_265833	
input
conv2d_26_265752
conv2d_26_265754!
batch_normalization_38_265757!
batch_normalization_38_265759!
batch_normalization_38_265761!
batch_normalization_38_265763
conv2d_27_265767
conv2d_27_265769!
batch_normalization_39_265772!
batch_normalization_39_265774!
batch_normalization_39_265776!
batch_normalization_39_265778
dense_38_265782
dense_38_265784
dense_39_265788
dense_39_265790!
batch_normalization_40_265793!
batch_normalization_40_265795!
batch_normalization_40_265797!
batch_normalization_40_265799
dense_40_265803
dense_40_265805
identity¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢.batch_normalization_40/StatefulPartitionedCall¢!conv2d_26/StatefulPartitionedCall¢!conv2d_27/StatefulPartitionedCall¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall
$average_pooling2d_39/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_39_layer_call_and_return_conditional_losses_2649362&
$average_pooling2d_39/PartitionedCallÎ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_39/PartitionedCall:output:0conv2d_26_265752conv2d_26_265754*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_26_layer_call_and_return_conditional_losses_2653362#
!conv2d_26/StatefulPartitionedCallÎ
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0batch_normalization_38_265757batch_normalization_38_265759batch_normalization_38_265761batch_normalization_38_265763*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_26538920
.batch_normalization_38/StatefulPartitionedCall·
$average_pooling2d_40/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_2650522&
$average_pooling2d_40/PartitionedCallÍ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_40/PartitionedCall:output:0conv2d_27_265767conv2d_27_265769*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_27_layer_call_and_return_conditional_losses_2654432#
!conv2d_27/StatefulPartitionedCallÍ
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0batch_normalization_39_265772batch_normalization_39_265774batch_normalization_39_265776batch_normalization_39_265778*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_26549620
.batch_normalization_39/StatefulPartitionedCall¶
$average_pooling2d_41/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_2651682&
$average_pooling2d_41/PartitionedCallÈ
 dense_38/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_41/PartitionedCall:output:0dense_38_265782dense_38_265784*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_2655692"
 dense_38/StatefulPartitionedCall
flatten_13/PartitionedCallPartitionedCall)dense_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_flatten_13_layer_call_and_return_conditional_losses_2655912
flatten_13/PartitionedCall¶
 dense_39/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_39_265788dense_39_265790*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2656152"
 dense_39/StatefulPartitionedCallÄ
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0batch_normalization_40_265793batch_normalization_40_265795batch_normalization_40_265797batch_normalization_40_265799*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_26530320
.batch_normalization_40/StatefulPartitionedCall
dropout_12/PartitionedCallPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_2656832
dropout_12/PartitionedCall¶
 dense_40/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_40_265803dense_40_265805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2657072"
 dense_40/StatefulPartitionedCallÂ
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_26_265752*'
_output_shapes
:*
dtype024
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:2%
#conv2d_26/kernel/Regularizer/Square¡
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_26/kernel/Regularizer/ConstÂ
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/Sum
"conv2d_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_26/kernel/Regularizer/mul/xÄ
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_26/kernel/Regularizer/mulÂ
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_27_265767*'
_output_shapes
:@*
dtype024
2conv2d_27/kernel/Regularizer/Square/ReadVariableOpÂ
#conv2d_27/kernel/Regularizer/SquareSquare:conv2d_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@2%
#conv2d_27/kernel/Regularizer/Square¡
"conv2d_27/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_27/kernel/Regularizer/ConstÂ
 conv2d_27/kernel/Regularizer/SumSum'conv2d_27/kernel/Regularizer/Square:y:0+conv2d_27/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/Sum
"conv2d_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752$
"conv2d_27/kernel/Regularizer/mul/xÄ
 conv2d_27/kernel/Regularizer/mulMul+conv2d_27/kernel/Regularizer/mul/x:output:0)conv2d_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_27/kernel/Regularizer/mul¶
1dense_38/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_38_265782*
_output_shapes

:@d*
dtype023
1dense_38/kernel/Regularizer/Square/ReadVariableOp¶
"dense_38/kernel/Regularizer/SquareSquare9dense_38/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@d2$
"dense_38/kernel/Regularizer/Square
!dense_38/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_38/kernel/Regularizer/Const¾
dense_38/kernel/Regularizer/SumSum&dense_38/kernel/Regularizer/Square:y:0*dense_38/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/Sum
!dense_38/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_38/kernel/Regularizer/mul/xÀ
dense_38/kernel/Regularizer/mulMul*dense_38/kernel/Regularizer/mul/x:output:0(dense_38/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_38/kernel/Regularizer/mul·
1dense_39/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_39_265788*
_output_shapes
:	Ð*
dtype023
1dense_39/kernel/Regularizer/Square/ReadVariableOp·
"dense_39/kernel/Regularizer/SquareSquare9dense_39/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Ð2$
"dense_39/kernel/Regularizer/Square
!dense_39/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_39/kernel/Regularizer/Const¾
dense_39/kernel/Regularizer/SumSum&dense_39/kernel/Regularizer/Square:y:0*dense_39/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/Sum
!dense_39/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752#
!dense_39/kernel/Regularizer/mul/xÀ
dense_39/kernel/Regularizer/mulMul*dense_39/kernel/Regularizer/mul/x:output:0(dense_39/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_39/kernel/Regularizer/mulÁ
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô

_user_specified_nameinput
Ê
¯
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_265120

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
c
²
__inference__traced_save_267317
file_prefix/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop;
7savev2_batch_normalization_38_gamma_read_readvariableop:
6savev2_batch_normalization_38_beta_read_readvariableopA
=savev2_batch_normalization_38_moving_mean_read_readvariableopE
Asavev2_batch_normalization_38_moving_variance_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop;
7savev2_batch_normalization_39_gamma_read_readvariableop:
6savev2_batch_normalization_39_beta_read_readvariableopA
=savev2_batch_normalization_39_moving_mean_read_readvariableopE
Asavev2_batch_normalization_39_moving_variance_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop;
7savev2_batch_normalization_40_gamma_read_readvariableop:
6savev2_batch_normalization_40_beta_read_readvariableopA
=savev2_batch_normalization_40_moving_mean_read_readvariableopE
Asavev2_batch_normalization_40_moving_variance_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop<
8savev2_sgd_conv2d_26_kernel_momentum_read_readvariableop:
6savev2_sgd_conv2d_26_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_38_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_38_beta_momentum_read_readvariableop<
8savev2_sgd_conv2d_27_kernel_momentum_read_readvariableop:
6savev2_sgd_conv2d_27_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_39_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_39_beta_momentum_read_readvariableop;
7savev2_sgd_dense_38_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_38_bias_momentum_read_readvariableop;
7savev2_sgd_dense_39_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_39_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_40_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_40_beta_momentum_read_readvariableop;
7savev2_sgd_dense_40_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_40_bias_momentum_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_dfa8a3c7ae49404881b873d10b29d25a/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename·
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*É
value¿B¼/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesæ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesö
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop7savev2_batch_normalization_38_gamma_read_readvariableop6savev2_batch_normalization_38_beta_read_readvariableop=savev2_batch_normalization_38_moving_mean_read_readvariableopAsavev2_batch_normalization_38_moving_variance_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop7savev2_batch_normalization_39_gamma_read_readvariableop6savev2_batch_normalization_39_beta_read_readvariableop=savev2_batch_normalization_39_moving_mean_read_readvariableopAsavev2_batch_normalization_39_moving_variance_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop7savev2_batch_normalization_40_gamma_read_readvariableop6savev2_batch_normalization_40_beta_read_readvariableop=savev2_batch_normalization_40_moving_mean_read_readvariableopAsavev2_batch_normalization_40_moving_variance_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop8savev2_sgd_conv2d_26_kernel_momentum_read_readvariableop6savev2_sgd_conv2d_26_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_38_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_38_beta_momentum_read_readvariableop8savev2_sgd_conv2d_27_kernel_momentum_read_readvariableop6savev2_sgd_conv2d_27_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_39_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_39_beta_momentum_read_readvariableop7savev2_sgd_dense_38_kernel_momentum_read_readvariableop5savev2_sgd_dense_38_bias_momentum_read_readvariableop7savev2_sgd_dense_39_kernel_momentum_read_readvariableop5savev2_sgd_dense_39_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_40_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_40_beta_momentum_read_readvariableop7savev2_sgd_dense_40_kernel_momentum_read_readvariableop5savev2_sgd_dense_40_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ã
_input_shapesÑ
Î: :::::::@:@:@:@:@:@:@d:d:	Ð:::::::: : : : : : : : :::::@:@:@:@:@d:d:	Ð:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@d: 

_output_shapes
:d:%!

_output_shapes
:	Ð: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::-#)
'
_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@: &

_output_shapes
:@:$' 

_output_shapes

:@d: (

_output_shapes
:d:%)!

_output_shapes
:	Ð: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
::$- 

_output_shapes

:: .

_output_shapes
::/

_output_shapes
: 
à
¯
$__inference_signature_wrapper_266183	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 **
f%R#
!__inference__wrapped_model_2649302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesv
t:ÿÿÿÿÿÿÿÿÿô::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô

_user_specified_nameinput
Æ
©
"__inference__traced_restore_267465
file_prefix%
!assignvariableop_conv2d_26_kernel%
!assignvariableop_1_conv2d_26_bias3
/assignvariableop_2_batch_normalization_38_gamma2
.assignvariableop_3_batch_normalization_38_beta9
5assignvariableop_4_batch_normalization_38_moving_mean=
9assignvariableop_5_batch_normalization_38_moving_variance'
#assignvariableop_6_conv2d_27_kernel%
!assignvariableop_7_conv2d_27_bias3
/assignvariableop_8_batch_normalization_39_gamma2
.assignvariableop_9_batch_normalization_39_beta:
6assignvariableop_10_batch_normalization_39_moving_mean>
:assignvariableop_11_batch_normalization_39_moving_variance'
#assignvariableop_12_dense_38_kernel%
!assignvariableop_13_dense_38_bias'
#assignvariableop_14_dense_39_kernel%
!assignvariableop_15_dense_39_bias4
0assignvariableop_16_batch_normalization_40_gamma3
/assignvariableop_17_batch_normalization_40_beta:
6assignvariableop_18_batch_normalization_40_moving_mean>
:assignvariableop_19_batch_normalization_40_moving_variance'
#assignvariableop_20_dense_40_kernel%
!assignvariableop_21_dense_40_bias 
assignvariableop_22_sgd_iter!
assignvariableop_23_sgd_decay)
%assignvariableop_24_sgd_learning_rate$
 assignvariableop_25_sgd_momentum
assignvariableop_26_total
assignvariableop_27_count
assignvariableop_28_total_1
assignvariableop_29_count_15
1assignvariableop_30_sgd_conv2d_26_kernel_momentum3
/assignvariableop_31_sgd_conv2d_26_bias_momentumA
=assignvariableop_32_sgd_batch_normalization_38_gamma_momentum@
<assignvariableop_33_sgd_batch_normalization_38_beta_momentum5
1assignvariableop_34_sgd_conv2d_27_kernel_momentum3
/assignvariableop_35_sgd_conv2d_27_bias_momentumA
=assignvariableop_36_sgd_batch_normalization_39_gamma_momentum@
<assignvariableop_37_sgd_batch_normalization_39_beta_momentum4
0assignvariableop_38_sgd_dense_38_kernel_momentum2
.assignvariableop_39_sgd_dense_38_bias_momentum4
0assignvariableop_40_sgd_dense_39_kernel_momentum2
.assignvariableop_41_sgd_dense_39_bias_momentumA
=assignvariableop_42_sgd_batch_normalization_40_gamma_momentum@
<assignvariableop_43_sgd_batch_normalization_40_beta_momentum4
0assignvariableop_44_sgd_dense_40_kernel_momentum2
.assignvariableop_45_sgd_dense_40_bias_momentum
identity_47¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9½
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*É
value¿B¼/B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesì
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ò
_output_shapes¿
¼:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_26_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_26_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2´
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_38_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3³
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_38_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4º
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_38_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¾
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_38_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_27_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_27_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8´
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_39_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_39_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¾
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_39_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Â
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_39_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_38_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_38_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_39_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_39_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¸
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_40_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17·
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_40_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¾
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_40_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Â
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_40_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20«
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_40_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_40_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22¤
AssignVariableOp_22AssignVariableOpassignvariableop_22_sgd_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¥
AssignVariableOp_23AssignVariableOpassignvariableop_23_sgd_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24­
AssignVariableOp_24AssignVariableOp%assignvariableop_24_sgd_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¨
AssignVariableOp_25AssignVariableOp assignvariableop_25_sgd_momentumIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¡
AssignVariableOp_26AssignVariableOpassignvariableop_26_totalIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¡
AssignVariableOp_27AssignVariableOpassignvariableop_27_countIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28£
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29£
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¹
AssignVariableOp_30AssignVariableOp1assignvariableop_30_sgd_conv2d_26_kernel_momentumIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31·
AssignVariableOp_31AssignVariableOp/assignvariableop_31_sgd_conv2d_26_bias_momentumIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Å
AssignVariableOp_32AssignVariableOp=assignvariableop_32_sgd_batch_normalization_38_gamma_momentumIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_sgd_batch_normalization_38_beta_momentumIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¹
AssignVariableOp_34AssignVariableOp1assignvariableop_34_sgd_conv2d_27_kernel_momentumIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35·
AssignVariableOp_35AssignVariableOp/assignvariableop_35_sgd_conv2d_27_bias_momentumIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Å
AssignVariableOp_36AssignVariableOp=assignvariableop_36_sgd_batch_normalization_39_gamma_momentumIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ä
AssignVariableOp_37AssignVariableOp<assignvariableop_37_sgd_batch_normalization_39_beta_momentumIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¸
AssignVariableOp_38AssignVariableOp0assignvariableop_38_sgd_dense_38_kernel_momentumIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¶
AssignVariableOp_39AssignVariableOp.assignvariableop_39_sgd_dense_38_bias_momentumIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¸
AssignVariableOp_40AssignVariableOp0assignvariableop_40_sgd_dense_39_kernel_momentumIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¶
AssignVariableOp_41AssignVariableOp.assignvariableop_41_sgd_dense_39_bias_momentumIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Å
AssignVariableOp_42AssignVariableOp=assignvariableop_42_sgd_batch_normalization_40_gamma_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ä
AssignVariableOp_43AssignVariableOp<assignvariableop_43_sgd_batch_normalization_40_beta_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¸
AssignVariableOp_44AssignVariableOp0assignvariableop_44_sgd_dense_40_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¶
AssignVariableOp_45AssignVariableOp.assignvariableop_45_sgd_dense_40_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_459
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÒ
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_46Å
Identity_47IdentityIdentity_46:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_47"#
identity_47Identity_47:output:0*Ï
_input_shapes½
º: ::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*°
serving_default
@
input7
serving_default_input:0ÿÿÿÿÿÿÿÿÿô<
dense_400
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:²×
·x
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Ç_default_save_signature
+È&call_and_return_all_conditional_losses
É__call__"ßs
_tf_keras_networkÃs{"class_name": "Functional", "name": "functional_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_27", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 500, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 5]}, "data_format": "channels_last"}, "name": "average_pooling2d_39", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["average_pooling2d_39", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_38", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_38", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 5]}, "data_format": "channels_last"}, "name": "average_pooling2d_40", "inbound_nodes": [[["batch_normalization_38", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["average_pooling2d_40", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_39", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_39", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 1]}, "data_format": "channels_last"}, "name": "average_pooling2d_41", "inbound_nodes": [[["batch_normalization_39", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_38", "inbound_nodes": [[["average_pooling2d_41", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_13", "inbound_nodes": [[["dense_38", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_39", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_40", "inbound_nodes": [[["dense_39", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_12", "inbound_nodes": [[["batch_normalization_40", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_40", "inbound_nodes": [[["dropout_12", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense_40", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 500, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_27", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 500, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 5]}, "data_format": "channels_last"}, "name": "average_pooling2d_39", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["average_pooling2d_39", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_38", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_38", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 5]}, "data_format": "channels_last"}, "name": "average_pooling2d_40", "inbound_nodes": [[["batch_normalization_38", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["average_pooling2d_40", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_39", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_39", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 1]}, "data_format": "channels_last"}, "name": "average_pooling2d_41", "inbound_nodes": [[["batch_normalization_39", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_38", "inbound_nodes": [[["average_pooling2d_41", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_13", "inbound_nodes": [[["dense_38", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_39", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_40", "inbound_nodes": [[["dense_39", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_12", "inbound_nodes": [[["batch_normalization_40", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_40", "inbound_nodes": [[["dropout_12", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense_40", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["categorical_accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 7.81249980263965e-07, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
õ"ò
_tf_keras_input_layerÒ{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 500, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 500, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}

regularization_losses
trainable_variables
	variables
	keras_api
+Ê&call_and_return_all_conditional_losses
Ë__call__"þ
_tf_keras_layerä{"class_name": "AveragePooling2D", "name": "average_pooling2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_39", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 5]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
®


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+Ì&call_and_return_all_conditional_losses
Í__call__"	
_tf_keras_layerí{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 100, 1]}}
À	
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+Î&call_and_return_all_conditional_losses
Ï__call__"ê
_tf_keras_layerÐ{"class_name": "BatchNormalization", "name": "batch_normalization_38", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_38", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 100, 256]}}

(regularization_losses
)trainable_variables
*	variables
+	keras_api
+Ð&call_and_return_all_conditional_losses
Ñ__call__"þ
_tf_keras_layerä{"class_name": "AveragePooling2D", "name": "average_pooling2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 5]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 5]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
°


,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
+Ò&call_and_return_all_conditional_losses
Ó__call__"	
_tf_keras_layerï{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 20, 256]}}
½	
2axis
	3gamma
4beta
5moving_mean
6moving_variance
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+Ô&call_and_return_all_conditional_losses
Õ__call__"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_39", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_39", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 20, 64]}}

;regularization_losses
<trainable_variables
=	variables
>	keras_api
+Ö&call_and_return_all_conditional_losses
×__call__"þ
_tf_keras_layerä{"class_name": "AveragePooling2D", "name": "average_pooling2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
·

?kernel
@bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+Ø&call_and_return_all_conditional_losses
Ù__call__"
_tf_keras_layerö{"class_name": "Dense", "name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 20, 64]}}
ê
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+Ú&call_and_return_all_conditional_losses
Û__call__"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
³

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
+Ü&call_and_return_all_conditional_losses
Ý__call__"
_tf_keras_layerò{"class_name": "Dense", "name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 20, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2000]}}
¶	
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
+Þ&call_and_return_all_conditional_losses
ß__call__"à
_tf_keras_layerÆ{"class_name": "BatchNormalization", "name": "batch_normalization_40", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_40", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
é
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
+à&call_and_return_all_conditional_losses
á__call__"Ø
_tf_keras_layer¾{"class_name": "Dropout", "name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ö

\kernel
]bias
^regularization_losses
_trainable_variables
`	variables
a	keras_api
+â&call_and_return_all_conditional_losses
ã__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
Ù
biter
	cdecay
dlearning_rate
emomentummomentum·momentum¸ momentum¹!momentumº,momentum»-momentum¼3momentum½4momentum¾?momentum¿@momentumÀImomentumÁJmomentumÂPmomentumÃQmomentumÄ\momentumÅ]momentumÆ"
	optimizer
@
ä0
å1
æ2
ç3"
trackable_list_wrapper

0
1
 2
!3
,4
-5
36
47
?8
@9
I10
J11
P12
Q13
\14
]15"
trackable_list_wrapper
Æ
0
1
 2
!3
"4
#5
,6
-7
38
49
510
611
?12
@13
I14
J15
P16
Q17
R18
S19
\20
]21"
trackable_list_wrapper
Î
fmetrics
regularization_losses
glayer_regularization_losses
hnon_trainable_variables
trainable_variables
ilayer_metrics
	variables

jlayers
É__call__
Ç_default_save_signature
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
-
èserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
kmetrics
regularization_losses
lnon_trainable_variables
trainable_variables

mlayers
nlayer_metrics
	variables
olayer_regularization_losses
Ë__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_26/kernel
:2conv2d_26/bias
(
ä0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
pmetrics
regularization_losses
qnon_trainable_variables
trainable_variables

rlayers
slayer_metrics
	variables
tlayer_regularization_losses
Í__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2batch_normalization_38/gamma
*:(2batch_normalization_38/beta
3:1 (2"batch_normalization_38/moving_mean
7:5 (2&batch_normalization_38/moving_variance
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
<
 0
!1
"2
#3"
trackable_list_wrapper
°
umetrics
$regularization_losses
vnon_trainable_variables
%trainable_variables

wlayers
xlayer_metrics
&	variables
ylayer_regularization_losses
Ï__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
zmetrics
(regularization_losses
{non_trainable_variables
)trainable_variables

|layers
}layer_metrics
*	variables
~layer_regularization_losses
Ñ__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
+:)@2conv2d_27/kernel
:@2conv2d_27/bias
(
å0"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
´
metrics
.regularization_losses
non_trainable_variables
/trainable_variables
layers
layer_metrics
0	variables
 layer_regularization_losses
Ó__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_39/gamma
):'@2batch_normalization_39/beta
2:0@ (2"batch_normalization_39/moving_mean
6:4@ (2&batch_normalization_39/moving_variance
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
µ
metrics
7regularization_losses
non_trainable_variables
8trainable_variables
layers
layer_metrics
9	variables
 layer_regularization_losses
Õ__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
;regularization_losses
non_trainable_variables
<trainable_variables
layers
layer_metrics
=	variables
 layer_regularization_losses
×__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
!:@d2dense_38/kernel
:d2dense_38/bias
(
æ0"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
µ
metrics
Aregularization_losses
non_trainable_variables
Btrainable_variables
layers
layer_metrics
C	variables
 layer_regularization_losses
Ù__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
Eregularization_losses
non_trainable_variables
Ftrainable_variables
layers
layer_metrics
G	variables
 layer_regularization_losses
Û__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
": 	Ð2dense_39/kernel
:2dense_39/bias
(
ç0"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
µ
metrics
Kregularization_losses
non_trainable_variables
Ltrainable_variables
layers
layer_metrics
M	variables
 layer_regularization_losses
Ý__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_40/gamma
):'2batch_normalization_40/beta
2:0 (2"batch_normalization_40/moving_mean
6:4 (2&batch_normalization_40/moving_variance
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
<
P0
Q1
R2
S3"
trackable_list_wrapper
µ
metrics
Tregularization_losses
non_trainable_variables
Utrainable_variables
layers
 layer_metrics
V	variables
 ¡layer_regularization_losses
ß__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¢metrics
Xregularization_losses
£non_trainable_variables
Ytrainable_variables
¤layers
¥layer_metrics
Z	variables
 ¦layer_regularization_losses
á__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
!:2dense_40/kernel
:2dense_40/bias
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
µ
§metrics
^regularization_losses
¨non_trainable_variables
_trainable_variables
©layers
ªlayer_metrics
`	variables
 «layer_regularization_losses
ã__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
0
¬0
­1"
trackable_list_wrapper
 "
trackable_list_wrapper
J
"0
#1
52
63
R4
S5"
trackable_list_wrapper
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
ä0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
å0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
æ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
ç0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
¿

®total

¯count
°	variables
±	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


²total

³count
´
_fn_kwargs
µ	variables
¶	keras_api"Ð
_tf_keras_metricµ{"class_name": "MeanMetricWrapper", "name": "categorical_accuracy", "dtype": "float32", "config": {"name": "categorical_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
®0
¯1"
trackable_list_wrapper
.
°	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
²0
³1"
trackable_list_wrapper
.
µ	variables"
_generic_user_object
6:42SGD/conv2d_26/kernel/momentum
(:&2SGD/conv2d_26/bias/momentum
6:42)SGD/batch_normalization_38/gamma/momentum
5:32(SGD/batch_normalization_38/beta/momentum
6:4@2SGD/conv2d_27/kernel/momentum
':%@2SGD/conv2d_27/bias/momentum
5:3@2)SGD/batch_normalization_39/gamma/momentum
4:2@2(SGD/batch_normalization_39/beta/momentum
,:*@d2SGD/dense_38/kernel/momentum
&:$d2SGD/dense_38/bias/momentum
-:+	Ð2SGD/dense_39/kernel/momentum
&:$2SGD/dense_39/bias/momentum
5:32)SGD/batch_normalization_40/gamma/momentum
4:22(SGD/batch_normalization_40/beta/momentum
,:*2SGD/dense_40/kernel/momentum
&:$2SGD/dense_40/bias/momentum
æ2ã
!__inference__wrapped_model_264930½
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *-¢*
(%
inputÿÿÿÿÿÿÿÿÿô
ò2ï
I__inference_functional_27_layer_call_and_return_conditional_losses_265748
I__inference_functional_27_layer_call_and_return_conditional_losses_266341
I__inference_functional_27_layer_call_and_return_conditional_losses_266472
I__inference_functional_27_layer_call_and_return_conditional_losses_265833À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_functional_27_layer_call_fn_265968
.__inference_functional_27_layer_call_fn_266521
.__inference_functional_27_layer_call_fn_266570
.__inference_functional_27_layer_call_fn_266102À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¸2µ
P__inference_average_pooling2d_39_layer_call_and_return_conditional_losses_264936à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
5__inference_average_pooling2d_39_layer_call_fn_264942à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266593¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_26_layer_call_fn_266602¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266622
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266686
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266640
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266704´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_38_layer_call_fn_266730
7__inference_batch_normalization_38_layer_call_fn_266666
7__inference_batch_normalization_38_layer_call_fn_266653
7__inference_batch_normalization_38_layer_call_fn_266717´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¸2µ
P__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_265052à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
5__inference_average_pooling2d_40_layer_call_fn_265058à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266753¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_27_layer_call_fn_266762¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266800
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266864
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266846
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266782´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
7__inference_batch_normalization_39_layer_call_fn_266826
7__inference_batch_normalization_39_layer_call_fn_266813
7__inference_batch_normalization_39_layer_call_fn_266890
7__inference_batch_normalization_39_layer_call_fn_266877´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¸2µ
P__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_265168à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
5__inference_average_pooling2d_41_layer_call_fn_265174à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
î2ë
D__inference_dense_38_layer_call_and_return_conditional_losses_266932¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_38_layer_call_fn_266941¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_13_layer_call_and_return_conditional_losses_266947¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_flatten_13_layer_call_fn_266952¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_39_layer_call_and_return_conditional_losses_266974¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_39_layer_call_fn_266983¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
â2ß
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_267019
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_267039´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¬2©
7__inference_batch_normalization_40_layer_call_fn_267065
7__inference_batch_normalization_40_layer_call_fn_267052´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_12_layer_call_and_return_conditional_losses_267077
F__inference_dropout_12_layer_call_and_return_conditional_losses_267082´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_12_layer_call_fn_267087
+__inference_dropout_12_layer_call_fn_267092´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
D__inference_dense_40_layer_call_and_return_conditional_losses_267103¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_40_layer_call_fn_267112¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
__inference_loss_fn_0_267123
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_267134
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_2_267145
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_3_267156
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
1B/
$__inference_signature_wrapper_266183input¬
!__inference__wrapped_model_264930 !"#,-3456?@IJSPRQ\]7¢4
-¢*
(%
inputÿÿÿÿÿÿÿÿÿô
ª "3ª0
.
dense_40"
dense_40ÿÿÿÿÿÿÿÿÿó
P__inference_average_pooling2d_39_layer_call_and_return_conditional_losses_264936R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ë
5__inference_average_pooling2d_39_layer_call_fn_264942R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
P__inference_average_pooling2d_40_layer_call_and_return_conditional_losses_265052R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ë
5__inference_average_pooling2d_40_layer_call_fn_265058R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿó
P__inference_average_pooling2d_41_layer_call_and_return_conditional_losses_265168R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ë
5__inference_average_pooling2d_41_layer_call_fn_265174R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266622 !"#N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266640 !"#N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266686t !"#<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿd
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿd
 Ê
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_266704t !"#<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿd
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿd
 Ç
7__inference_batch_normalization_38_layer_call_fn_266653 !"#N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
7__inference_batch_normalization_38_layer_call_fn_266666 !"#N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_38_layer_call_fn_266717g !"#<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿd
p
ª "!ÿÿÿÿÿÿÿÿÿd¢
7__inference_batch_normalization_38_layer_call_fn_266730g !"#<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "!ÿÿÿÿÿÿÿÿÿdÈ
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266782r3456;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 È
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_266800r3456;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 í
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2668463456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 í
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2668643456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
  
7__inference_batch_normalization_39_layer_call_fn_266813e3456;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@ 
7__inference_batch_normalization_39_layer_call_fn_266826e3456;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@Å
7__inference_batch_normalization_39_layer_call_fn_2668773456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Å
7__inference_batch_normalization_39_layer_call_fn_2668903456M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¸
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_267019bRSPQ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_267039bSPRQ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_40_layer_call_fn_267052URSPQ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_40_layer_call_fn_267065USPRQ3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¶
E__inference_conv2d_26_layer_call_and_return_conditional_losses_266593m7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿd
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿd
 
*__inference_conv2d_26_layer_call_fn_266602`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿd
ª "!ÿÿÿÿÿÿÿÿÿd¶
E__inference_conv2d_27_layer_call_and_return_conditional_losses_266753m,-8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_27_layer_call_fn_266762`,-8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@´
D__inference_dense_38_layer_call_and_return_conditional_losses_266932l?@7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿd
 
)__inference_dense_38_layer_call_fn_266941_?@7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿd¥
D__inference_dense_39_layer_call_and_return_conditional_losses_266974]IJ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÐ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_dense_39_layer_call_fn_266983PIJ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÐ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_40_layer_call_and_return_conditional_losses_267103\\]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_40_layer_call_fn_267112O\]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dropout_12_layer_call_and_return_conditional_losses_267077\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
F__inference_dropout_12_layer_call_and_return_conditional_losses_267082\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dropout_12_layer_call_fn_267087O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ~
+__inference_dropout_12_layer_call_fn_267092O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ«
F__inference_flatten_13_layer_call_and_return_conditional_losses_266947a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿd
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÐ
 
+__inference_flatten_13_layer_call_fn_266952T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿÐÎ
I__inference_functional_27_layer_call_and_return_conditional_losses_265748 !"#,-3456?@IJRSPQ\]?¢<
5¢2
(%
inputÿÿÿÿÿÿÿÿÿô
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Î
I__inference_functional_27_layer_call_and_return_conditional_losses_265833 !"#,-3456?@IJSPRQ\]?¢<
5¢2
(%
inputÿÿÿÿÿÿÿÿÿô
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
I__inference_functional_27_layer_call_and_return_conditional_losses_266341 !"#,-3456?@IJRSPQ\]@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿô
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
I__inference_functional_27_layer_call_and_return_conditional_losses_266472 !"#,-3456?@IJSPRQ\]@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿô
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
.__inference_functional_27_layer_call_fn_265968s !"#,-3456?@IJRSPQ\]?¢<
5¢2
(%
inputÿÿÿÿÿÿÿÿÿô
p

 
ª "ÿÿÿÿÿÿÿÿÿ¥
.__inference_functional_27_layer_call_fn_266102s !"#,-3456?@IJSPRQ\]?¢<
5¢2
(%
inputÿÿÿÿÿÿÿÿÿô
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¦
.__inference_functional_27_layer_call_fn_266521t !"#,-3456?@IJRSPQ\]@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿô
p

 
ª "ÿÿÿÿÿÿÿÿÿ¦
.__inference_functional_27_layer_call_fn_266570t !"#,-3456?@IJSPRQ\]@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿô
p 

 
ª "ÿÿÿÿÿÿÿÿÿ;
__inference_loss_fn_0_267123¢

¢ 
ª " ;
__inference_loss_fn_1_267134,¢

¢ 
ª " ;
__inference_loss_fn_2_267145?¢

¢ 
ª " ;
__inference_loss_fn_3_267156I¢

¢ 
ª " ¸
$__inference_signature_wrapper_266183 !"#,-3456?@IJSPRQ\]@¢=
¢ 
6ª3
1
input(%
inputÿÿÿÿÿÿÿÿÿô"3ª0
.
dense_40"
dense_40ÿÿÿÿÿÿÿÿÿ