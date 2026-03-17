terraform {
	required_version = ">= 1.9.0"

	required_providers {
		azurerm = {
			source  = "hashicorp/azurerm"
			version = "~> 4.0"
		}
	}
}

provider "azurerm" {
	features {}
}

# -------------------------
# Basic configuration
# -------------------------

locals {
	prefix        = "mlops-vm"        # Used to name resources
	location      = "switzerlandnorth"   # Azure region
}

# -------------------------
# Resource group
# -------------------------

resource "azurerm_resource_group" "rg" {
	location = local.location
	name     = "${local.prefix}-rg"
}

# -------------------------
# Networking
# -------------------------

resource "azurerm_virtual_network" "vnet" {
	address_space       = ["10.0.0.0/16"]
	location            = azurerm_resource_group.rg.location
	name                = "${local.prefix}-vnet"
	resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_subnet" "subnet" {
	address_prefixes     = ["10.0.1.0/24"]
	name                 = "${local.prefix}-subnet"
	resource_group_name  = azurerm_resource_group.rg.name
	virtual_network_name = azurerm_virtual_network.vnet.name
}

resource "azurerm_network_security_group" "nsg" {
	location            = azurerm_resource_group.rg.location
	name                = "${local.prefix}-nsg"
	resource_group_name = azurerm_resource_group.rg.name
}

# Allow SSH from anywhere (you can restrict source later)
resource "azurerm_network_security_rule" "ssh" {
	access                      = "Allow"
	direction                   = "Inbound"
	name                        = "SSH"
	network_security_group_name = azurerm_network_security_group.nsg.name
	priority                    = 100
	protocol                    = "Tcp"
	resource_group_name         = azurerm_resource_group.rg.name
	source_address_prefix       = "*"
	source_port_range           = "*"
	destination_address_prefix  = "*"
	destination_port_range      = "22"
}

resource "azurerm_network_security_rule" "api_http" {
	access                      = "Allow"
	direction                   = "Inbound"
	name                        = "API_HTTP_8000"
	network_security_group_name = azurerm_network_security_group.nsg.name
	priority                    = 110
	protocol                    = "Tcp"
	resource_group_name         = azurerm_resource_group.rg.name
	source_address_prefix       = "*"
	source_port_range           = "*"
	destination_address_prefix  = "*"
	destination_port_range      = "8000"
}

resource "azurerm_network_security_rule" "locust_ui" {
	access                      = "Allow"
	direction                   = "Inbound"
	name                        = "LOCUST_UI_8089"
	network_security_group_name = azurerm_network_security_group.nsg.name
	priority                    = 120
	protocol                    = "Tcp"
	resource_group_name         = azurerm_resource_group.rg.name
	source_address_prefix       = "*"
	source_port_range           = "*"
	destination_address_prefix  = "*"
	destination_port_range      = "8089"
}

resource "azurerm_public_ip" "pip" {
	allocation_method   = "Static"
	location            = azurerm_resource_group.rg.location
	name                = "${local.prefix}-pip"
	resource_group_name = azurerm_resource_group.rg.name
	sku                 = "Standard"
}

resource "azurerm_network_interface" "nic" {
	location            = azurerm_resource_group.rg.location
	name                = "${local.prefix}-nic"
	resource_group_name = azurerm_resource_group.rg.name

	ip_configuration {
		name                          = "primary"
		private_ip_address_allocation = "Dynamic"
		public_ip_address_id          = azurerm_public_ip.pip.id
		subnet_id                     = azurerm_subnet.subnet.id
	}
}

resource "azurerm_network_interface_security_group_association" "nic_nsg" {
	network_interface_id      = azurerm_network_interface.nic.id
	network_security_group_id = azurerm_network_security_group.nsg.id
}

# -------------------------
# Linux VM
# -------------------------

resource "azurerm_linux_virtual_machine" "vm" {
	admin_username                  = "mlopsadmin" # change if you want
	disable_password_authentication = true
	location                        = azurerm_resource_group.rg.location
	name                            = "${local.prefix}-vm"
	network_interface_ids           = [azurerm_network_interface.nic.id]
	resource_group_name             = azurerm_resource_group.rg.name
	size                            = "Standard_B2s" # small & cheap for tests

	admin_ssh_key {
		public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdUv9+/6FFh3uroeRtfCLnBb6PIcIfxi1FwGzsfOFE5 mlops-omist" # TODO: put your real SSH public key here
		username   = "mlopsadmin"
	}

	os_disk {
		caching              = "ReadWrite"
		storage_account_type = "Standard_LRS"
	}

	source_image_reference {
		offer     = "0001-com-ubuntu-server-jammy"
		publisher = "Canonical"
		sku       = "22_04-lts-gen2"
		version   = "latest"
	}

	computer_name = "${local.prefix}-vm"
}

# -------------------------
# Outputs (for remote access)
# -------------------------

output "vm_public_ip" {
	description = "Public IP of the VM"
	value       = azurerm_public_ip.pip.ip_address
}