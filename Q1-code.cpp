#include<iostream>
#include<math.h>
#include<time.h>
using namespace std;

#define e 2.71828

double a[6] = { 0 };
double weight1[6] = {0.29,-0.83,-0.09,-0.19,0.93,0.9};
double weight2[6] = {0.14,0.72,0.65,0.16,-0.11,0.43};
double weight3[6] = {-0.51,0.08,0.72,0.65,0.62,0.14};
double weight4[6] = {-0.7,-0.77,0.6,0.18,-0.69,0.82};
double weight5[6] = { -0.8,-0.91,0.93,0.98,0.09,0.57 };
double l;
double te;

class ANN
{
public:
	double w11;
	double w12;
	double w21;
	double w22;
	double w1;
	double w2;
	double o1;
	double o2;
	double o3;

	double delw11;
	double delw12;
	double delw21;
	double delw22;
	double delw1;
	double delw2;

	ANN()     // class constructor
	{
		
			int x = rand() % 5;	

			if (x == 0)
			{
				w1 = weight1[0];
				w2 = weight1[1];
				w11 = weight1[2];
				w12 = weight1[3];
				w21 = weight1[4];
				w22 = weight1[5];
				
			}
			else if (x == 1)
			{
				w1 = weight2[0];
				w2 = weight2[1];
				w11 = weight2[2];
				w12 = weight2[3];
				w21 = weight2[4];
				w22 = weight2[5];
			}
			else if (x == 2)
			{
				w1 = weight3[0];
				w2 = weight3[1];
				w11 = weight3[2];
				w12 = weight3[3];
				w21 = weight3[4];
				w22 = weight3[5];
			}
			else if (x == 3)
			{
				w1 = weight4[0];
				w2 = weight4[1];
				w11 = weight4[2];
				w12 = weight4[3];
				w21 = weight4[4];
				w22 = weight4[5];
			}
			else if (x == 4)
			{
				w1 = weight5[0];
				w2 = weight5[1];
				w11 = weight5[2];
				w12 = weight5[3];
				w21 = weight5[4];
				w22 = weight5[5];
			}

		delw1 = 0;
		delw2 = 0;
		delw11 = 0;
		delw12 = 0;
		delw21 = 0;
		delw22 = 0;

	}

	double train(int x1,int x2,int x3)   // train function
	{
		double y1, y2, y3 = 0;
		y1 = x1*w11 + x2*w12;
		y2 = x1*w21 + x2*w22;
		o1 = 1/(1+pow(e, -y1));
		o2 = 1 / (1+pow(e, -y2));

		y3 = o1*w1 + o2*w2;
		o3= 1 / (1+pow(e, -y3));     //output 

	

		delw1 =delw1+ l*(x3 - o3)*o3*(1 - o3)*o1;
		delw2 = delw2+l*(x3 - o3)*o3*(1 - o3)*o2;   //calculate weight value

		double a = (x3 - o3)*o3*(1 - o3);

		delw11 = delw11 + l*x1*a*w1*o1*(1 - o1);
		delw12 = delw12 + l*x2*a*w1*o1*(1 - o1);


		delw21 = delw21 + l*x1*a*w2*o2*(1 - o2);
		delw22 = delw22 + l*x2*a*w2*o2*(1 - o2);     //calculate weight value

		double error = 0.5*(x3 - o3)*(x3 - o3);
		return error;       //return error
	}

	void updata()    //function to updata
	{
		w1 = w1+delw1;
		w2 = w2+delw2;
		w11 = w11+delw11;
		w12 = w12+delw12;
		w21 = w21+delw21;
		w22 = w22+delw22;

		delw1 = 0;
		delw2 = 0;
		delw11 = 0;
		delw12 = 0;
		delw21 = 0;
		delw22 = 0;
	}
};


int main()
{
	srand(time(0));

	ANN net;
	
	cout << "Please input the learning rate:" << endl;
	cin >> l;
	cout << "Please input the target error:" << endl;
	cin >> te;


	cout << "The initial weights is:" << endl;
	cout << net.w1 << endl;
	cout << net.w2 << endl;
	cout << net.w11 << endl;
	cout << net.w12 << endl;
	cout << net.w21 << endl;
	cout << net.w22 << endl;


	double err;
	err=net.train(0,0,0);
	err=err+ net.train(0, 1, 1);
	err = err + net.train(1, 0, 1);
	err = err + net.train(1, 1, 0); //calculate error

	net.updata();          // update weight

	cout << endl;
	cout <<"The error of first batch is "<< err << endl;

	int i = 0;
	while (err > te)
	{
		err = net.train(0, 0, 0);
		err = err + net.train(0, 1, 1);
		err = err + net.train(1, 0, 1);
		err = err + net.train(1, 1, 0);

		net.updata();
		//cout << i << "  " << err<<endl;
		i++;
	}    

	cout << endl;
	cout << "The final weights is:" << endl;
	cout << net.w1 << endl;
	cout << net.w2 << endl;
	cout << net.w11 << endl;
	cout << net.w12 << endl;
	cout << net.w21 << endl;
	cout << net.w22 << endl;
	cout << endl;

	cout << "The final error is " << err<<endl;

	cout << "The total number of batches is " << i << endl;


	return 0;
}