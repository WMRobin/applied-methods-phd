using CSV
using DataFrames
using LinearAlgebra
using Distributions

cd("homework\\homework1");

nsw = CSV.read("lalonde_nsw.csv", DataFrame);

re78_treat = nsw["re78"] .* nsw["treat"];
re78_control = nsw["re78"] .* (1 .- nsw["treat"]);
N = size(nsw,1);
N_0 = sum(nsw[:treat]);
N_1 = sum(1 .- nsw[:treat]);

#########################
####################  A #
#########################
ATE = (1/N) * (nsw[:treat]' * nsw[:re78] - (1 .- nsw[:treat])' * nsw[:re78]);

#########################
####################  B #
#########################
ATT = (1/N_1) * nsw[:treat]' * nsw[:re78] - (1 .- nsw[:treat])' * nsw[:re78];

@info "Treatment effects" ATE ATT ATE-ATT

#########################
####################  C #
#########################
re78_treat = Vector{Float64}()
re78_control = Vector{Float64}()
using Random
using StatsBase

# permute tretament vector
perm_treatvec = StatsBase.sample(nsw[:treat],1000*size(nsw,1))
perm_mat = [perm_treatvec repeat(nsw[:re78], 1000)];

# two sample t test
ATE_perm = (1/size(perm_mat,1)) * (perm_treatvec' * perm_mat[:,2] - (1 .- perm_treatvec)' * perm_mat[:,2])
ATT_perm = (1/sum(perm_treatvec)) * perm_treatvec' * perm_mat[:,2] - (1/(1000*size(nsw,1) - sum(perm_treatvec))) * (1 .- perm_treatvec)' * perm_mat[:,2];


#########################
####################  D #
#########################
using GLM
lm(@formula(re78 ~ treat), nsw)


#########################
############ QUESTION 2 #
#########################
psid = CSV.read("lalonde_psid.csv", DataFrame)

# combine treated nsw and constrol psid
psid_nsw = [psid; nsw[nsw.treat .== 1,:]]

# estimate propensity score
logit = glm(@formula(treat ~ age + education + black + married + nodegree + re74 + re75), psid_nsw, Binomial(), LogitLink())
psid_nsw[:pscore] = predict(logit, psid_nsw);
describe(psid_nsw)
gdfscore = DataFrame(treat = psid_nsw.treat, pscore = psid_nsw.pscore); gdfscore = groupby(gdfscore, :treat);
combine(gdfscore, nrow, vec(valuecols(gdfscore) .=> [mean std]))

using Plots
using StatsPlots
treat_pscore = psid_nsw.pscore[psid_nsw.treat .== 1]
control_pscore = psid_nsw.pscore[psid_nsw.treat .== 0]
density(treat_pscore)
density(control_pscore)

Y = psid_nsw.re78; D = psid_nsw.treat; pi_X = psid_nsw.pscore;
tau_ipw = (1/size(psid_nsw,1)) * ((Y .* D) ./ pi_X .- (Y .* (1.-D)))



psid_nsw.treat' * psid_nsw.re78 / psid_nsw.pscore 