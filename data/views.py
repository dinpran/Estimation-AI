from django.shortcuts import render,HttpResponse
from .model import CO2EmissionModel,BreastCancerModel,HousePricePredictionModel,ChurnPredictionModel,FuelConsumptionModel,YachtModel
import ssl
from urllib.request import urlopen

# Create your views here.
def index(request):
    return render(request,'index.html')

def emissions(request):                             
    return render(request, 'emissions.html')

def cancer(request):
    return render(request, 'cancer.html')

def estimation(request):
    return render(request, 'estimation.html')


def churn(request):
    return render(request, 'churn.html')

def consumption(request):                             
    return render(request, 'consumption.html')

def about(request):
    return render(request, 'about.html')

def contact_us(request):
    return render(request, 'contact_us.html')

def price(request):
    return render(request, 'price.html')

def predict_inputs(request):
    if request.method == 'POST':
        try:
            engine_size = float(request.POST.get('1st_input'))
            cylinders = float(request.POST.get('2nd_input'))

            # Predict the CO2 emissions
            model = CO2EmissionModel()
            result = model.predict(engine_size, cylinders)

        except (ValueError, TypeError) as e:
            result = f"Invalid input. Error: {str(e)}"

        return render(request, 'x_result.html', {'result': result})
    return HttpResponse("Invalid request method.")


def predict_cancer(request):
    if request.method == 'POST':
        try:
            radius_mean = float(request.POST.get('1st_input'))
            texture_mean = float(request.POST.get('2nd_input'))
            perimeter_mean = float(request.POST.get('3rd_input'))
            area_mean = float(request.POST.get('4th_input'))
            

            # Predict the cancer diagnosis
            model = BreastCancerModel()
            result = model.predict(radius_mean, texture_mean, perimeter_mean, area_mean)

            if result == 1:
                diagnosis = "Malignant"
            else:
                diagnosis = "Benign"

        except (ValueError, TypeError) as e:
            diagnosis = f"Invalid input. Error: {str(e)}"

        return render(request, 'y_result.html', {'diagnosis': diagnosis})
    return HttpResponse("Invalid request method.")

def predict_house_price(request):
    if request.method == 'POST':
        try:
            # Retrieve and validate input data
            crim = float(request.POST.get('CRIM_input'))
            zn = float(request.POST.get('ZN_input'))
            indus = float(request.POST.get('INDUS_input'))
            chas = float(request.POST.get('CHAS_input'))
            nox = float(request.POST.get('NOX_input'))
            rm = float(request.POST.get('RM_input'))
            age = float(request.POST.get('AGE_input'))
            dis = float(request.POST.get('DIS_input'))
            rad = float(request.POST.get('RAD_input'))
            tax = float(request.POST.get('TAX_input'))
            ptratio = float(request.POST.get('PTRATIO_input'))
            lstat = float(request.POST.get('LSTAT_input'))

            # Predict the house price
            model = HousePricePredictionModel()
            predicted_price = model.predict(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, lstat)

            result = f"Predicted House Price: ${predicted_price:.2f}"

        except (ValueError, TypeError) as e:
            result = f"Invalid input. Error: {str(e)}"

        return render(request, 'house_price_result.html', {'result': result})
    return HttpResponse("Invalid request method.")

def predict_churn(request):
    if request.method == 'POST':
        try:
            tenure = float(request.POST.get('tenure_input'))
            age = float(request.POST.get('age_input'))
            address = float(request.POST.get('address_input'))
            income = float(request.POST.get('income_input'))
            ed = float(request.POST.get('ed_input'))
            employ = float(request.POST.get('employ_input'))
            equip = float(request.POST.get('equip_input'))

            model = ChurnPredictionModel()
            result = model.predict(tenure, age, address, income, ed, employ, equip)

            if result == 1:
                prediction = "Churn"
            else:
                prediction = "Not Churn"

        except (ValueError, TypeError) as e:
            prediction = f"Invalid input. Error: {str(e)}"

        return render(request, 'churn_result.html', {'prediction': prediction})
    return HttpResponse("Invalid request method.")

def predict_consumption(request):
    if request.method == 'POST':
        try:
            engine_size = float(request.POST.get('1st_input'))
            cylinders = float(request.POST.get('2nd_input'))
            co2_emission = float(request.POST.get('3nd_input'))

            # Predict the CO2 emissions
            model = FuelConsumptionModel()
            result = model.predict(engine_size, cylinders,co2_emission)

        except (ValueError, TypeError) as e:
            result = f"Invalid input. Error: {str(e)}"

        return render(request, 'consumption_result.html', {'result': result})
    return HttpResponse("Invalid request method.")

def yacht(request):
    return render(request, 'yacht.html')

def predict_yacht_hydrodynamics(request):
    if request.method == 'POST':
        try:
            # Retrieve user input
            long_position = float(request.POST.get('long_position'))
            prismatic_coefficient = float(request.POST.get('prismatic_coefficient'))
            length_displacement_ratio = float(request.POST.get('length_displacement_ratio'))
            bean_draught_ratio = float(request.POST.get('bean_draught_ratio'))
            length_bean_ratio = float(request.POST.get('length_bean_ratio'))
            froude_number = float(request.POST.get('froude_number'))

            # Load the model and perform prediction
            model = YachtModel()  # Create an instance of the model
            predicted_resistance = model.predict(long_position, prismatic_coefficient, length_displacement_ratio, bean_draught_ratio, length_bean_ratio, froude_number)

            result = f"Predicted Residuary Resistance: {predicted_resistance:.2f}"

        except (ValueError, TypeError) as e:
            result = f"Invalid input. Error: {str(e)}"

        return render(request, 'yacht_result.html', {'result': result})
    return HttpResponse("Invalid request method.")



