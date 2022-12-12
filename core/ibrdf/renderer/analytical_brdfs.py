import numpy as np
import torch
import torch.nn as nn

merl_mask = torch.load('/'.join(__file__.split('/')[:-1])+'/merl_mask.pt').float()

# theta_i : BS * 1
# R0      : BS * 1
# ret     : BS * 1
def Fresnel(theta_i, R0):
    return R0 + (1 - R0) * torch.clamp((1 - torch.cos(theta_i)), 0, 1)**5

# x      : BS * 3(theta_h,theta_d,phi_d)
# params : BS * x
# ret    : BS * 3(rgb)
def CosThetaIn(x, params):
    theta_h = x[:,0:1]
    theta_d = x[:,1:2]
    phi_d = x[:,2:3]

    wi = torch.cat(
        [
        (torch.cos(theta_h)*torch.sin(theta_d)*torch.cos(phi_d)+torch.sin(theta_h)*torch.cos(theta_d)), 
        (torch.sin(theta_d)*torch.sin(phi_d)), 
        (-torch.sin(theta_h)*torch.sin(theta_d)*torch.cos(phi_d)+torch.cos(theta_h)*torch.cos(theta_d))
        ], 
        dim=1
      )

    n = torch.cat(
        [
        torch.zeros_like(theta_h), 
        torch.zeros_like(theta_h), 
        torch.ones_like(theta_h)
        ], 
        dim=1
      )

    cos_theta_i = torch.clamp(torch.sum(wi*n, dim=1, keepdim=True),0,1)
    return cos_theta_i.repeat(1,3) # BS*3

# x      : BS * 3(theta_h,theta_d,phi_d)
# params : BS * x
# ret    : BS * 3(rgb)
def Phong(x, params):
    diffuse_color = params[:,:3]
    specular_color = params[:,3:6]
    specular_exp = params[:,6:7]
    
    theta_h = x[:,0:1]
    theta_d = x[:,1:2]
    phi_d = x[:,2:3]

    wi = torch.cat(
        [
        (torch.cos(theta_h)*torch.sin(theta_d)*torch.cos(phi_d)+torch.sin(theta_h)*torch.cos(theta_d)), 
        (torch.sin(theta_d)*torch.sin(phi_d)), 
        (-torch.sin(theta_h)*torch.sin(theta_d)*torch.cos(phi_d)+torch.cos(theta_h)*torch.cos(theta_d))
        ], 
        dim=1
      )
    half = torch.cat(
        [
        torch.sin(theta_h), 
        torch.zeros_like(theta_h), 
        torch.cos(theta_h)
        ], 
        dim=1
      )
    n = torch.cat(
        [
        torch.zeros_like(theta_h), 
        torch.zeros_like(theta_h), 
        torch.ones_like(theta_h)
        ], 
        dim=1
      )
    
    wo = 2*torch.sum(wi*half, dim=1, keepdim=True)*half - wi
    wr = 2*torch.sum(wi*n, dim=1, keepdim=True)*n - wi

    theta_i = torch.acos(torch.clamp(torch.sum(wi*n, dim=1, keepdim=True),-1,1))
    theta_o = torch.acos(torch.clamp(torch.sum(wo*n, dim=1, keepdim=True),-1,1))
    theta_r = torch.acos(torch.clamp(torch.sum(wo*wr, dim=1, keepdim=True),-1,1))

    diffuse_brdf = 1.0 / (2.0*np.pi) * diffuse_color
    # workaroung to ensure Helmholtz reciprocity
    specular_brdf1 = (specular_exp+1)/(2*np.pi)*torch.clamp(torch.cos(theta_r),1e-6,1)**specular_exp / torch.clamp(torch.cos(theta_i), 1e-4, 1) * specular_color
    specular_brdf2 = (specular_exp+1)/(2*np.pi)*torch.clamp(torch.cos(theta_r),1e-6,1)**specular_exp / torch.clamp(torch.cos(theta_o), 1e-4, 1) * specular_color
    specular_brdf = 0.5 * (specular_brdf1 + specular_brdf2)
    specular_brdf = specular_brdf * 0.5*(torch.sign(torch.cos(theta_r)) + 1)
    return diffuse_brdf + specular_brdf # BS*3



def Lambert(x, params):
    diffuse_color = params[:,:3]
    return 1.0 / (2.0*np.pi) * diffuse_color

def TrowbridgeReitzMicrofacetBRDF(x, params):
    color = params[:,:3]
    alpha = params[:,3:4]
    R0 = params[:,4:5]
    
    theta_h = x[:,0:1]
    theta_d = x[:,1:2]
    phi_d = x[:,2:3]

    wi = torch.cat(
        [
        (torch.cos(theta_h)*torch.sin(theta_d)*torch.cos(phi_d)+torch.sin(theta_h)*torch.cos(theta_d)), 
        (torch.sin(theta_d)*torch.sin(phi_d)), 
        (-torch.sin(theta_h)*torch.sin(theta_d)*torch.cos(phi_d)+torch.cos(theta_h)*torch.cos(theta_d))
        ], 
        dim=1
      )
    half = torch.cat(
        [
        torch.sin(theta_h), 
        torch.zeros_like(theta_h), 
        torch.cos(theta_h)
        ], 
        dim=1
      )
    n = torch.cat(
        [
        torch.zeros_like(theta_h), 
        torch.zeros_like(theta_h), 
        torch.ones_like(theta_h)
        ], 
        dim=1
      )
    
    wo = 2*torch.sum(wi*half, dim=1, keepdim=True)*half - wi
    
    theta_i = torch.acos(torch.clamp(torch.sum(wi*n, dim=1, keepdim=True),-1,1))
    theta_o = torch.acos(torch.clamp(torch.sum(wo*n, dim=1, keepdim=True),-1,1))
    
    phi_o = torch.atan2(wo[:,1:2], wo[:,0:1]) - torch.atan2(wo[:,1:2], wo[:,0:1])
    phi_i = torch.atan2(wi[:,1:2], wi[:,0:1]) - torch.atan2(wo[:,1:2], wo[:,0:1])

    tmp = 1 + torch.tan(theta_h)**2 / alpha**2
    denom = np.pi * alpha**2 * torch.cos(theta_h)**4 * tmp**2
    D = 1.0 / denom

    def geo_lambda(theta, phi, alpha_x, alpha_y):
        absTanTheta = torch.abs(torch.tan(torch.clamp(theta, 0.0, 0.5*0.999*np.pi)))
        alpha = torch.sqrt(torch.cos(phi)**2 * (alpha_x**2) + torch.sin(phi) * (alpha_y**2))
        return (-1 + torch.sqrt(1 + (alpha * absTanTheta)**2)) / 2    

    G = 1 / (1 + geo_lambda(theta_i, phi_i, alpha, alpha) + geo_lambda(theta_o, phi_o, alpha, alpha))

    F = Fresnel(theta_d, R0)
    
    cos_theta_i = torch.clamp(torch.cos(theta_i), 1e-3, 1)
    cos_theta_o = torch.clamp(torch.cos(theta_o), 1e-3, 1)
    return color * D * G * F / (4 * cos_theta_i * cos_theta_o)

kBRDFSamplingResThetaH = 90
kBRDFSamplingResThetaD = 90
kBRDFSamplingResPhiD = 360

kRedScale = 1.0 / 1500.0
kGreenScale = 1.15 / 1500.0
kBlueScale = 1.66 / 1500.0

# params : BS
def generateMERL(model, params):
    idx1,idx2,idx3 = torch.meshgrid(
        torch.arange(kBRDFSamplingResThetaH, dtype=params.dtype, device=params.device), 
        torch.arange(kBRDFSamplingResThetaD, dtype=params.dtype, device=params.device), 
        torch.arange(kBRDFSamplingResPhiD//2, dtype=params.dtype, device=params.device)
    )
    idx1 = idx1 + 0.5
    idx2 = idx2 + 0.5
    idx3 = idx3 + 0.5

    theta_h = idx1**2 / (kBRDFSamplingResThetaH**2) * (np.pi / 2.0)
    theta_d = idx2 / kBRDFSamplingResThetaD * (np.pi * 0.5)
    phi_d = idx3 / (kBRDFSamplingResPhiD / 2) * np.pi
    x = torch.cat([theta_h[:,:,:,None], theta_d[:,:,:,None], phi_d[:,:,:,None]], dim=3)
    x = x.view(-1,3) # BS * 3
    params = params[None,:].expand(x.size(0),params.size(0))

    brdf = model(x, params)
    brdf = brdf * merl_mask.to(brdf.device).view(-1,1)

    # scale
    brdf_coef = 1.0 / torch.tensor([kRedScale,kGreenScale,kBlueScale], dtype=params.dtype, device=params.device)
    brdf = brdf * brdf_coef[None,:]

    brdf = brdf.transpose(0,1)
    return brdf.view(3, kBRDFSamplingResThetaH, kBRDFSamplingResThetaD, kBRDFSamplingResPhiD//2)