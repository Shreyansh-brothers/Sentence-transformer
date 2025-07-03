#include <iostream>

using namespace std;

int main() {
    int a;
    double b;
    
    cout << "Enter first number: "<< endl;
    cin >> a;

    cout << "Enter the second number: "<< endl;
    cin >> b;

    cout << "Sum is: "<< (a+b)<<"\n";
    cout << "Difference is: "<< (a-b)<<"\n";
    cout << "Product is: "<< (a*b)<<"\n";
    cout << "Division is: "<< (a/b)<<"\n";
    cout << "Modulo is: "<< (a%(int)b)<<"\n";
    return 0;
}
