#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

using namespace std;
int test_cpu();
void test_gpu();
void test_inference();

int main(int argc, char *argv[])
{
    // torch::Device device(torch::kCUDA);
    // auto array = torch::rand(10);
    // std::cout << dev_arr << std::endl;
    // auto dev_arr = array.to(device);
    // std::cout << dev_arr << std::endl;

    try
    {
        if (torch::cuda::is_available())
            test_gpu();
        else
            test_cpu();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Caught an exception : " << e.what() << '\n';
    }

    return 0;
}

// Define a new Module.
struct Net : torch::nn::Module
{
    Net()
    {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int test_cpu()
{
    cout << "Run test on CPU" << endl;
    // Create a new Net.
    auto net = std::make_shared<Net>();

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/64);

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch)
    {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 100 == 0)
            {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
            }
        }
    }

    return 0;
}

void test_gpu()
{
    cout << "Run test on GPU" << endl;
    torch::Device device(torch::kCUDA);

    // Create a new Net.
    auto net = std::make_shared<Net>();
    net->to(device);

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/64);

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch)
    {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            // Reset gradients.
            optimizer.zero_grad();

            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(batch.data.to(device));

            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nll_loss(prediction, batch.target.to(device));

            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 100 == 0)
            {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
            }
        }
    }
}

void test_inference()
{
    torch::Device device(torch::kCUDA);
    // Deserialize the ScriptModule from a file using torch::jit::load()
    torch::jit::script::Module module = torch::jit::load("model.pt");
    module.to(device);

    // Create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}).to(device));

    // Exectute the model
    at::Tensor output = module.forward(inputs).toTensor();

    std::cout << output.slice(/*dims=*/1, /*start=*/0, /*end=*/5) << '\n';

    std::cout << "ok\n";
}