from django.forms import ModelForm
from django import forms
from faceApp.models import regData



class RegForm(ModelForm):
		class Meta:
			model = regData
			fields = ['name', 'employee_id']
			widgets={
			'name':forms.TextInput(attrs={
				'class':'form-control',
				'placeholder':'Name'
				}),

			'employee_id':forms.TextInput(attrs={
				'class':'form-control',
				'placeholder':'Employee_id'
				})
			}