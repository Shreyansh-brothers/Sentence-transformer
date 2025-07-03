#include <iostream>

using namespace std;

int Binarytodecimal(int n){

    int ans = 0, pow = 1;
    while (n > 0)
    {
        int rem = n % 10;
        ans += rem * pow;
        n /= 10;
        pow *= 2;
    }
    return ans;
}


int main()
{
    int a;
    cout << "Enter the number " << endl;
    cin >> a;
    cout << "Decimal form of given number is: " << Binarytodecimal(a) << endl;

    return 0;
}

