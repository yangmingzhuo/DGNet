% 
% denoised = load('denoised.mat');
% gt = load('clean.mat');

% denoised = denoised.denoised;
% gt = gt.clean;

total_psnr = 0;
total_ssim = 0;
for i = 1:length(denoised)
   denoised_patch = squeeze(denoised(i,:,:,:));
   gt_patch = squeeze(clean(i,:,:,:));
   ssim_val = ssim(denoised_patch, gt_patch);
   psnr_val = psnr(denoised_patch, gt_patch);
   fprintf('%d PSNR: %f SSIM: %f\n', i, psnr_val, ssim_val);
   total_ssim = total_ssim + ssim_val;
   total_psnr = total_psnr + psnr_val;
end
qm_psnr = total_psnr / length(denoised);
qm_ssim = total_ssim / length(denoised);

fprintf('PSNR: %f SSIM: %f\n', qm_psnr, qm_ssim);