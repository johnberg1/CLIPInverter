from PIL import Image
import matplotlib.pyplot as plt
import torch

# Log images
def log_image(x, opts):
	return tensor2im(x)

# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or any([param is item for item in blacklist]) or param.grad == None:
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)

def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

# This is to plot the cycle too
def vis_faces(log_hooks, txt, mismatch_text):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(12, 3 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	plt.gcf().text(0.02, 0.89, 'Original: {}'.format(txt[0]), fontsize=8)
	if mismatch_text:
		plt.gcf().text(0.02, 0.49, 'Mismatch: {}'.format(txt[-1]), fontsize=8)
		plt.gcf().text(0.02, 0.01, 'Mismatch: {}'.format(txt[0]), fontsize=8)
	else:
		plt.gcf().text(0.02, 0.49, 'Mismatch: {}'.format(txt[0]), fontsize=8)
		plt.gcf().text(0.02, 0.01, 'Mismatch: {}'.format(txt[1]), fontsize=8)
	plt.gcf().text(0.02, 0.41, 'Original: {}'.format(txt[1]), fontsize=8)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		vis_faces_with_text(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig

def vis_faces_frozen(log_hooks, txt, mismatch_text):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(12, 4 * display_count))
	gs = fig.add_gridspec(display_count, 4)
	plt.gcf().text(0.02, 0.52, 'Original: {}'.format(txt[0]), fontsize=8)
	if mismatch_text:
		plt.gcf().text(0.02, 0.49, 'Mismatch: {}'.format(txt[-1]), fontsize=8)
		plt.gcf().text(0.02, 0.01, 'Mismatch: {}'.format(txt[0]), fontsize=8)
	else:
		plt.gcf().text(0.02, 0.49, 'Mismatch: {}'.format(txt[0]), fontsize=8)
		plt.gcf().text(0.02, 0.01, 'Mismatch: {}'.format(txt[1]), fontsize=8)
	plt.gcf().text(0.02, 0.04, 'Original: {}'.format(txt[1]), fontsize=8)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		vis_faces_with_frozen(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig
 


# This is to plot the cycle too
def vis_faces_with_text(hooks_dict, fig, gs, i):
	fig.add_subplot(gs[i, 0])
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}\n'.format(float(hooks_dict['diff_input'])))
	plt.axis('off')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\nTarget Sim={:.2f}\n'.format(float(hooks_dict['diff_input'])))
	plt.axis('off')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['recovered_face'])
	plt.title('Recovered\n')
	plt.axis('off')

def vis_faces_with_frozen(hooks_dict, fig, gs, i):
	fig.add_subplot(gs[i, 0])
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['frozen_face'])
	plt.title('Frozen')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Trained')
	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['recovered_face'])
	plt.title('Recovered')

    
