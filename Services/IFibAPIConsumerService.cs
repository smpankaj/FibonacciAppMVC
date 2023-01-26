namespace FibonacciAppMVC.Services
{
    public interface IFibAPIConsumerService
    {
        public Task<string> consumeAPIAsync(string num);

    }
}
